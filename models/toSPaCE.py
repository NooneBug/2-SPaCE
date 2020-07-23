from dataloaders.ShimaokaDataLoader import ShimaokaDataset

from models.input_encoder_models.ShimaokaModel import ShimaokaMentionAndContextEncoder

from models.CommonNetwork import CommonNetwork

from models.projectors.multiProjectorManager import MultiProjectorManager 
from models.projectors.NickelProjector import NickelProjector

from models.classifier_module import Classifier

from losses.MultiLossManager import MultiLossManager

from optimizers.riemannianAdamOptimizer import RiemannianAdamOptimizer

from torch.nn import Module

from torch.utils.data import DataLoader

from evaluation_routines.evaluators import RegressionEvaluator, ClassifierEvaluator

import torch
import numpy as np
from tqdm import tqdm

class ComposedRegressiveNetwork(Module):

  def __init__(self, config, word_lookup, type_lookup, foldername):
    
    super().__init__()

    self.config = config
    self.define_factory()
    self.set_parameters()
    self.initialize_loss_manager()
    self.folder_name = foldername

    self.word_lookup = word_lookup

    self.word_network = self.get_network_class('WORD_MANIPULATION_MODULE')(config)

    self.type_lookup = type_lookup

    self.common_module = self.get_network_class('DEEP_NETWORK')(config)

    self.projectors = self.get_network_class('PROJECTORS')(config)

  def set_parameters(self):
    self.epochs = int(self.config['TRAINING_PARAMETERS']['epochs'])
    self.early_stopping = bool(self.config['TRAINING_PARAMETERS']['early_stopping'])
    self.early_stopping_trigger = False
    self.patience = int(self.config['TRAINING_PARAMETERS']['patience'])
  
  def set_optimizer(self, config):
    nametag = 'TRAINING_PARAMETERS'

    optimizer_config = config[nametag]['optimizer']
    optimizer_class = self.optimizer_factory[config[optimizer_config]['class']]
    
    self.optimizer = optimizer_class(config, self)

  def get_network_class(self, config_key):
    return self.network_factory[self.config[self.config['2-SPACE MODULES CONFIGS'][config_key]]['CLASS']]

  def define_factory(self):
    self.network_factory = {
      "ShimaokaModel" : ShimaokaMentionAndContextEncoder,
      "CommonNetwork": CommonNetwork,
      'MultiProjectorsManager': MultiProjectorManager
    }
    
    self.dataLoader_factory = {
      "ShimaokaDataLoader": ShimaokaDataset
    }

    self.losses_factory = {
      'MultiLossManager': MultiLossManager
    }

    self.optimizer_factory = {
      'riemannianAdamOptimizer': RiemannianAdamOptimizer
    }
  
  def get_DataLoader(self, dataset, batch_size, shuffle = False):
    dl = self.dataLoader_factory[self.config[self.config['2-SPACE MODULES CONFIGS']['WORD_MANIPULATION_MODULE']]['DATALOADER_CLASS']](dataset)

    return DataLoader(dl, batch_size=batch_size , shuffle=shuffle)

  def forward(self, input):
    contexts, positions, context_len = input[0], input[1], input[2]
    mentions, mention_chars = input[3], input[4]
    type_indexes = input[5]

    words_batch = self.word_lookup(contexts)

    mention_vec = self.word_network.mention_encoder(mentions, mention_chars, self.word_lookup)
    context_vec, _ = self.word_network.context_encoder(words_batch, positions, context_len, self.word_lookup)

    input_vec = torch.cat((mention_vec, context_vec), dim=1)
    # print(mention_vec)
    # print('mention shape: {}'.format(mention_vec.shape))
    # print('context shape: {}'.format(context_vec.shape))
    # print('input shape: {}'.format(input_vec.shape))

    common_output = self.common_module(input_vec)

    # print('common_output shape: {}'.format(common_output.shape))

    projections = self.projectors(common_output)

    # print('projetions keys: {}'.format([k for k in projections]))
    # print('projections shapes: {}'.format([p.shape for p in projections.values()]))

    return {'projections': projections}

  def train_(self, train_loader, val_loader):
    
    loss_SUM = 0
    val_loss_SUM = 0
    total_examples = 0
    total_val_examples = 0

    e = 0
    while e < self.epochs and not self.early_stopping_trigger:
      bar = tqdm(total = len(train_loader), desc='{}^ epoch: training'.format(e + 1))
      for data in train_loader:
        self.optimizer.zero_grad()
        self.train()

        model_output = self(data)

        # print('model_output: {}'.format(model_output))

        labels = data[5]
        true_vectors = self.get_true_vectors(labels)

        loss = self.compute_loss(model_output, true_vectors)

        loss = self.compute_loss_value(loss)
        loss.backward()

        loss_SUM += loss.item()
        total_examples += len(data)
        
        bar.update(1)

        self.optimizer.step()

      with torch.no_grad():
        self.eval()
        bar.close()
        bar = tqdm(total = len(val_loader), desc='{}^ epoch: validation'.format(e + 1))
        for data in val_loader:  
        
          model_output = self(data)

          labels = data[5]
          true_vectors = self.get_true_vectors(labels)

          val_loss = self.compute_loss(model_output, true_vectors)

          val_loss = self.compute_loss_value(val_loss)

          val_loss_SUM += val_loss
          total_val_examples += len(data)
          bar.update(1)

        bar.close()
      
      self.print_stats(loss_SUM, val_loss_SUM, total_examples, total_val_examples)
      self.early_stopping_routine(value = val_loss_SUM, epoch = e)
      e += 1


    return self          

  def print_stats(self, loss_value, val_loss_value, train_examples, val_examples):
    print('\t train loss: {:.4f}\t val loss: {:.4f}\n'.format(loss_value/train_examples, 
                                                            val_loss_value/val_examples))
  
  def early_stopping_routine(self, value, epoch):
    if self.early_stopping:
      if epoch == 0:
        self.min_val_loss = value
        self.best_epoch = epoch
        self.save_model(epoch)
      elif value <= self.min_val_loss:
        self.best_epoch = epoch
        self.save_model(epoch)
      elif self.best_epoch + self.patience < epoch:
        print('EarlyStopping')
        self.early_stopping_trigger = True
    print('\t best epoch: {}\n'.format(self.best_epoch))
  
  def save_model(self, epoch):
    torch.save({
                'model_state_dict' : self.state_dict(),
                'epoch' : epoch 
              }, 
              self.folder_name + '/model.pth')
    

  def compute_loss_value(self, loss):
    mean_losses = [torch.mean(v) for v in loss.values()]

    loss_sum = sum(mean_losses)

    return loss_sum


  def get_true_vectors(self, labels):
    true_vectors = {}

    for k, lookup in self.type_lookup.items():
      true_vectors[k] = lookup(labels)
      
    return true_vectors

  def initialize_loss_manager(self):
    loss_keyword = self.config['2-SPACE MODULES CONFIGS']['LOSS']
    
    loss_config = self.config[loss_keyword]['Class']

    self.loss_manager = self.losses_factory[loss_config](self.config)

  def compute_loss(self, projectors_output, true_vectors):
    '''projectors_output are batched projectors outputs'''
    

    return self.loss_manager.compute_loss(true_vectors, projectors_output['projections'])

  def evaluate(self, testLoader, folderName):
    test_pred, true_labels = self.get_test_predictions_and_labels(testLoader)

    evaluator = RegressionEvaluator(predictions= test_pred, 
                              labels = true_labels, 
                              type_lookup = self.type_lookup)

    evaluator.evaluate(folderName)

  def get_test_predictions_and_labels(self, testLoader):
    all_labels = []
    for i, data in enumerate(testLoader):
      pred = self(data)
      print('pred: {}'.format(pred))
      if i == 0:
        all_predictions = {k: v.detach().cpu().numpy() for k, v in pred['projections'].items()}
      else:
        pred = self(data)
        for k, v in pred.items():
          all_predictions[k].extend(v.detach().cpu().numpy())
      all_labels.extend(data[5].detach().cpu().numpy())
    return all_predictions, np.array(all_labels)


class ComposedClassificationNetwork(ComposedRegressiveNetwork):

  def __init__(self, config, word_lookup, type_lookup, foldername):
    super().__init__(config, word_lookup, type_lookup, foldername)

    self.nametag = 'CLASSIFIER'
    self.setup_factory()

    # print('type lookup: {}'.format(type_lookup))

    self.classifier = self.factory[config[self.nametag]['CLASS']](config, 
                                                                  self.get_types_number(type_lookup))

  def setup_factory(self):
    self.factory = {
      'Classifier': Classifier
    }

  def get_types_number(self, type_lookup):
    return max([emb.embedding_number for emb in type_lookup.values()])

  def forward(self, input):
    contexts, positions, context_len = input[0], input[1], input[2]
    mentions, mention_chars = input[3], input[4]
    type_indexes = input[5]

    words_batch = self.word_lookup(contexts)

    mention_vec = self.word_network.mention_encoder(mentions, mention_chars, self.word_lookup)
    context_vec, _ = self.word_network.context_encoder(words_batch, positions, context_len, self.word_lookup)

    input_vec = torch.cat((mention_vec, context_vec), dim=1)
    
    # print('mention shape: {}'.format(mention_vec.shape))
    # print('context shape: {}'.format(context_vec.shape))
    # print('input shape: {}'.format(input_vec.shape))

    common_output = self.common_module(input_vec)

    # print('common_output shape: {}'.format(common_output.shape))

    projections = self.projectors(common_output)

    # print('projetions keys: {}'.format([k for k in projections]))
    # print('projections shapes: {}'.format([p.shape for p in projections.values()]))

    projections_concat = torch.cat(tuple([p for p in projections.values()]), dim=-1)

    classifier_input = torch.cat((common_output, projections_concat), dim = -1)

    # print('classifier input shape: {}'.format(classifier_input.shape))

    classifier_output = self.classifier(classifier_input)

    # print('classifier output shape: {}'.format(classifier_output.shape))

    return {'projections': projections, 'classifier_output': classifier_output}

  def train_(self, train_loader, val_loader):
    loss_SUM = 0
    val_loss_SUM = 0

    regression_loss_SUM = 0
    classifier_loss_SUM = 0
    
    regression_val_loss_SUM = 0
    classifier_val_loss_SUM = 0
    
    total_examples = 0
    total_val_examples = 0

    e = 0
    while e < self.epochs and not self.early_stopping_trigger:
      bar = tqdm(total = len(train_loader), desc='{}^ epoch: training'.format(e + 1))
      for data in train_loader: 

        self.optimizer.zero_grad()
        self.train()

        model_output = self(data)

        labels = data[5]
        true_vectors = self.get_true_vectors(labels)

        loss = self.compute_loss(model_output, true_vectors)

        OH_labels = self.get_OneHot_labels(labels)

        classifier_loss = self.compute_classifier_loss(model_output, OH_labels)

        total_loss = self.compute_loss_value(loss, classifier_loss)
        
        # print('loss: {}'.format(total_loss))
        # print('classifier_loss: {}'.format(classifier_loss))
        
        total_loss.backward()

        self.optimizer.step()

        # regression_loss_SUM += loss
        classifier_loss_SUM += classifier_loss

        loss_SUM += total_loss.item()
        total_examples += len(data)

        bar.update(1)
      with torch.no_grad():
        self.eval()
        bar.close()

        bar = tqdm(total = len(val_loader), desc='{}^ epoch: validation'.format(e + 1))
        for data in val_loader:  
        
          model_output = self(data)

          labels = data[5]
          true_vectors = self.get_true_vectors(labels)

          val_loss = self.compute_loss(model_output, true_vectors)

          OH_labels = self.get_OneHot_labels(labels)

          classifier_loss = self.compute_classifier_loss(model_output, OH_labels)

          # print('loss: {}'.format(loss))
          # print('classifier_loss: {}'.format(classifier_loss))

          total_val_loss = self.compute_loss_value(val_loss, classifier_loss)

          # regression_val_loss_SUM += val_loss
          classifier_val_loss_SUM += classifier_loss
          
          val_loss_SUM += total_val_loss
          total_val_examples += len(data)


          bar.update(1)

      bar.close()

      self.print_stats(loss_SUM, val_loss_SUM, 
                      # regression_loss_SUM, regression_val_loss_SUM,
                      classifier_loss_SUM, classifier_val_loss_SUM,	
                      total_examples, total_val_examples)
      self.early_stopping_routine(value = val_loss_SUM, epoch = e)
      e += 1

    return self

  def print_stats(self, loss_SUM, val_loss_SUM, 
                        # regression_loss_SUM, regression_val_loss_SUM,
                        classifier_loss_SUM, classifier_val_loss_SUM,	
                        total_examples, total_val_examples):

    super().print_stats(loss_SUM, val_loss_SUM, total_examples, total_val_examples)

    # print('\tregression_loss: {:.4f}\t regression_val_loss: {:.4f}\n'.format(regression_loss_SUM/total_examples,
                                                                            # regression_val_loss_SUM/total_val_examples))

    print('\tclassifeir_loss: {:.4f}\t classifier_val_loss: {:.4f}\n'.format(classifier_loss_SUM/total_examples,
                                                                            classifier_val_loss_SUM, total_val_examples))

  def compute_loss_value(self, loss, classifier_loss):
    mean_losses = [torch.mean(v) for v in loss.values()]

    loss_sum = sum(mean_losses)

    return loss_sum.add_(classifier_loss)

  def get_OneHot_labels(self, labels):
    n_classes = self.get_types_number(self.type_lookup)
    
    onehot_labels = torch.zeros((len(labels), n_classes)).cuda()
    
    ones = torch.ones((len(labels), n_classes)).cuda()


    out = onehot_labels.scatter_(-1, labels, ones).cuda()

    return out

  # def compute_loss(self, regression_output, true_vectors):
  #   loss_keyword = self.config['2-SPACE MODULES CONFIGS']['LOSS']
  #   loss_config = self.config[loss_keyword]['Class']
  #   loss_manager = self.losses_factory[loss_config](self.config)

  #   return loss_manager.compute_loss(true_vectors, regression_output)

  def compute_classifier_loss(self, classifier_output, true_labels):
    return self.classifier.compute_loss(classifier_output['classifier_output'], true_labels)

  def evaluate(self, testLoader, folderName):
    test_vector_pred, test_classifier_pred, true_labels = self.get_test_predictions_and_labels(testLoader)

    evaluator = RegressionEvaluator(predictions= test_vector_pred, 
                              labels = true_labels, 
                              type_lookup = self.type_lookup)


    evaluator.evaluate(folderName)

    evaluator = ClassifierEvaluator(predictions = test_classifier_pred,
                                    labels = true_labels,
                                    type_lookup = self.type_lookup)
    
    evaluator.evaluate(folderName)

  def get_test_predictions_and_labels(self, testLoader):
    all_labels = []
    for i, data in enumerate(testLoader):
      pred = self(data)
      # print('pred: {}'.format(pred))
      if i == 0:
        all_vector_predictions = {k: v.detach().cpu().numpy() for k, v in pred['projections'].items()}
        all_classifier_predictions = [pred['classifier_output'].detach().cpu().numpy()]
      else:
        for k, v in pred['projections'].items():
          all_vector_predictions[k].extend(v.detach().cpu().numpy())
        all_classifier_predictions.extend(pred['classifier_output']).detach().cpu().numpy()
      all_labels.extend(data[5].detach().cpu().numpy())
    return all_vector_predictions, all_classifier_predictions[0], np.array(all_labels)

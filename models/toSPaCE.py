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

import torch

class ComposedRegressiveNetwork(Module):

  def __init__(self, config, word_lookup, type_lookup):
    
    super().__init__()

    self.config = config
    self.define_factory()
    self.set_parameters()
    self.initialize_loss_manager()

    self.word_lookup = word_lookup

    self.word_network = self.get_network_class('WORD_MANIPULATION_MODULE')(config)

    self.type_lookup = type_lookup

    self.common_module = self.get_network_class('DEEP_NETWORK')(config)

    self.projectors = self.get_network_class('PROJECTORS')(config)

  def set_parameters(self):
    self.epochs = int(self.config['TRAINING_PARAMETERS']['epochs'])
  
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

    return projections

  def train_(self, train_loader, val_loader):
    
    loss_SUM = 0
    val_loss_SUM = 0

    for e in range(self.epochs):
      for data in train_loader:  
        
        
        self.optimizer.zero_grad()
        self.train()

        model_output = self(data)

        # print('model_output: {}'.format(model_output))

        labels = data[5]
        true_vectors = self.get_true_vectors(labels)

        loss = self.compute_loss(model_output, true_vectors)

        loss = self.compute_loss_value(loss)
        print('loss: {}'.format(loss))
        loss.backward()

        loss_SUM += loss.item()

        self.optimizer.step()

      with torch.no_grad():
        self.eval()

        for data in val_loader:  
        
          model_output = self(data)

          labels = data[5]
          true_vectors = self.get_true_vectors(labels)

          val_loss = self.compute_loss(model_output, true_vectors)

          val_loss = self.compute_loss_value(val_loss)

          val_loss_SUM += val_loss

          

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
    

    return self.loss_manager.compute_loss(true_vectors, projectors_output)

class ComposedClassificationNetwork(ComposedRegressiveNetwork):

  def __init__(self, config, word_lookup, type_lookup):
    super().__init__(config, word_lookup, type_lookup)

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

    return projections, classifier_output

  def train_(self, train_loader, val_loader):
    loss_SUM = 0
    val_loss_SUM = 0

    for e in range(self.epochs):
      print('epoch: {}'.format(e + 1))
      for data in train_loader: 

        self.optimizer.zero_grad()
        self.train()

        regression_output, classifier_output = self(data)

        labels = data[5]
        true_vectors = self.get_true_vectors(labels)

        loss = self.compute_loss(regression_output, true_vectors)

        OH_labels = self.get_OneHot_labels(labels)

        classifier_loss = self.compute_classifier_loss(classifier_output, OH_labels)

        total_loss = self.compute_loss_value(loss, classifier_loss)
        
        print('loss: {}'.format(total_loss))
        print('classifier_loss: {}'.format(classifier_loss))
        
        total_loss.backward()

        self.optimizer.step()

        loss_SUM += total_loss.item()

      with torch.no_grad():
        self.eval()

        for data in val_loader:  
        
          model_output, classifier_output = self(data)

          labels = data[5]
          true_vectors = self.get_true_vectors(labels)

          val_loss = self.compute_loss(model_output, true_vectors)

          OH_labels = self.get_OneHot_labels(labels)

          classifier_loss = self.compute_classifier_loss(classifier_output, OH_labels)

          # print('loss: {}'.format(loss))
          # print('classifier_loss: {}'.format(classifier_loss))

          val_loss = self.compute_loss_value(val_loss, classifier_loss)

          val_loss_SUM += val_loss

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

  def compute_loss(self, regression_output, true_vectors):
    loss_keyword = self.config['2-SPACE MODULES CONFIGS']['LOSS']
    loss_config = self.config[loss_keyword]['Class']
    loss_manager = self.losses_factory[loss_config](self.config)

    return loss_manager.compute_loss(true_vectors, regression_output)

  def compute_classifier_loss(self, classifier_output, true_labels):
    return self.classifier.compute_loss(classifier_output, true_labels)
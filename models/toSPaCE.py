from dataloaders.ShimaokaDataLoader import ShimaokaDataset
from models.input_encoder_models.ShimaokaModel import ShimaokaMentionAndContextEncoder
from models.CommonNetwork import CommonNetwork
from models.projectors.multiProjectorHandler import MultiProjectorHandler 
from models.projectors.NickelProjector import NickelProjector
from models.classifier_module import Classifier
from torch.nn import Module
from torch.utils.data import DataLoader
import torch



class ComposedRegressiveNetwork(Module):

  def __init__(self, config, word_lookup, type_lookup):
    
    super().__init__()

    self.config = config
    self.define_factory()

    self.word_lookup = word_lookup

    self.word_network = self.get_network_class('WORD_MANIPULATION_MODULE')(config)

    self.type_lookup = type_lookup

    self.common_module = self.get_network_class('DEEP_NETWORK')(config)

    self.projectors = self.get_network_class('PROJECTORS')(config)
  
  def get_network_class(self, config_key):
    return self.network_factory[self.config[self.config['2-SPACE MODULES CONFIGS'][config_key]]['CLASS']]

  def define_factory(self):
    self.network_factory = {
      "ShimaokaModel" : ShimaokaMentionAndContextEncoder,
      "CommonNetwork": CommonNetwork,
      'MultiProjectorsHandler': MultiProjectorHandler
    }
    self.dataLoader_factory = {
      "ShimaokaDataLoader": ShimaokaDataset
    }
  
  def get_DataLoader(self, dataset, batch_size, shuffle):
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
    print('mention shape: {}'.format(mention_vec.shape))
    print('context shape: {}'.format(context_vec.shape))
    print('input shape: {}'.format(input_vec.shape))

    common_output = self.common_module(input_vec)

    print('common_output shape: {}'.format(common_output.shape))

    projections = self.projectors(common_output)

    print('projections shapes: {}'.format([p.shape for p in projections]))

    return projections

  def train_(self, dataloader):
    
    for data in dataloader:  
      loss = self(data)


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
    
    print('mention shape: {}'.format(mention_vec.shape))
    print('context shape: {}'.format(context_vec.shape))
    print('input shape: {}'.format(input_vec.shape))

    common_output = self.common_module(input_vec)

    print('common_output shape: {}'.format(common_output.shape))

    projections = self.projectors(common_output)

    print('projections shapes: {}'.format([p.shape for p in projections]))

    projections_concat = torch.cat(tuple([p for p in projections]), dim=-1)

    classifier_input = torch.cat((common_output, projections_concat), dim = -1)

    print('classifier input shape: {}'.format(classifier_input.shape))

    classifier_output = self.classifier(classifier_input)

    print('classifier output shape: {}'.format(classifier_output.shape))

    return projections, classifier_output

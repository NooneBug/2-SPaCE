from dataloaders.ShimaokaDataLoader import ShimaokaDataset
from models.input_encoder_models.ShimaokaModel import ShimaokaMentionAndContextEncoder
from models.CommonNetwork import CommonNetwork
from torch.nn import Module
from torch.utils.data import DataLoader
import torch


class ComposedNetwork(Module):

  def __init__(self, config, word_lookup, type_lookup):
    
    super().__init__()

    self.config = config
    self.define_factory()

    self.word_lookup = word_lookup

    self.word_network = self.get_network_class('WORD_MANIPULATION_MODULE')(config)

    self.type_lookup = type_lookup

    self.common_module = self.get_network_class('DEEP_NETWORK')(config)

  
  def get_network_class(self, config_key):
    return self.network_factory[self.config[self.config['2-SPACE MODULES CONFIGS'][config_key]]['CLASS']]

  def define_factory(self):
    self.network_factory = {
      "ShimaokaModel" : ShimaokaMentionAndContextEncoder,
      "CommonNetwork": CommonNetwork
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

    return True

  def train_(self, dataloader):
    
    for data in dataloader:  
      loss = self(data)

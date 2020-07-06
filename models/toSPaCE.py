from dataloaders.ShimaokaDataLoader import ShimaokaDataset
from models.input_encoder_models.ShimaokaModel import ShimaokaMentionAndContextEncoder
from torch.nn import Module
from torch.utils.data import DataLoader



class ComposedNetwork(Module):

  def __init__(self, config, word_lookup, type_lookup):
    
    self.config = config
    self.define_factory()

    self.word_lookup = word_lookup

    self.word_network = self.network_factory[config[config['2-SPACE MODULES CONFIGS']['WORD_MANIPULATION_MODULE']]['CLASS']](config)

    self.type_lookup = type_lookup

  def define_factory(self):
    self.network_factory = {
      "ShimaokaModel" : ShimaokaMentionAndContextEncoder
    }
    self.dataLoader_factory = {
      "ShimaokaDataLoader": ShimaokaDataset
    }
  
  def get_DataLoader(self, dataset, batch_size, shuffle):
    dl = self.dataLoader_factory[self.config[self.config['2-SPACE MODULES CONFIGS']['WORD_MANIPULATION_MODULE']]['DATALOADER_CLASS']](dataset)

    return DataLoader(dl, batch_size=batch_size , shuffle=shuffle)

  def forward(self, input):
    pass
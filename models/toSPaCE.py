from dataloaders.ShimaokaDataLoader import ShimaokaDataset
from models.input_encoder_models.ShimaokaModel import ShimaokaMentionAndContextEncoder
from models.CommonNetwork import CommonNetwork
from torch.nn import Module
from torch.utils.data import DataLoader



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
    pass

  def train_(self, dataloader):
    pass

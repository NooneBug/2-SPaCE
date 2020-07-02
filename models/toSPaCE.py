
from models.input_encoder_models.ShimaokaModel import ShimaokaMentionAndContextEncoder
from torch.nn import Module


class ComposedNetwork(Module):

  def __init__(self, config, word_lookup, type_lookup):
    
    self.define_factory()

    self.word_lookup = word_lookup

    self.word_network = self.network_factory[config[config['2-SPACE MODULES CONFIGS']['WORD_MANIPULATION_MODULE']]['CLASS']](config)

    self.type_lookup = type_lookup

  def define_factory(self):
    self.network_factory = {
      "ShimaokaModel" : ShimaokaMentionAndContextEncoder
    }
  
  def forward(self, input):
    pass
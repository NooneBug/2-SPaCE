from torch import nn
from torch.nn import Module
from models.geooptModules import MobiusLinear, mobius_linear, create_ball


class NickelProjector(Module):

  def __init__(self, config):
    self.nametag = 'HYPERBOLIC_PROJECTOR'
    self.conf = dict(config[self.nametag])
    self.cast_params()

    super().__init__()
    self.layers = nn.ModuleList()
    
    self.relu = nn.ReLU().cuda()

    prec = self.input_dim
        
    for dim in self.layers_dims:
      self.layers.append(MobiusLinear(prec, dim).cuda())
            
      prec = dim

  def cast_params(self):
    self.layers_dims = [int(splitted) for splitted in self.conf['layers'].split(' ')]
    self.input_dim = int(self.conf['input_size'])

  def forward(self, x):
    for i in range(len(self.layers) - 1):
        x = self.leaky_relu(self.layers[i](x))
    layers = self.layers[-1](x)
    return layers
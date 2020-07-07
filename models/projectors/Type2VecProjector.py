from torch.nn import Module
from torch import nn

class Type2VecProjector(Module):

  def __init__(self, config):
    self.nametag = 'COSINE_PROJECTOR'
    self.conf = dict(config[self.nametag])
    self.cast_params()

    super().__init__()
    self.layers = nn.ModuleList()
    
    self.relu = nn.ReLU().cuda()

    prec = self.input_dim
        
    for dim in self.layers_dims:
      self.layers.append(nn.Linear(prec, dim).cuda())
            
      prec = dim

  def cast_params(self):
    self.layers_dims = [int(splitted) for splitted in self.conf['layers'].split(' ')]
    self.input_dim = int(self.conf['input_size'])

  def forward(self, x):
    for i in range(len(self.layers) - 1):
        x = self.relu(self.layers[i](x))
    layers = self.layers[-1](x)
    return layers
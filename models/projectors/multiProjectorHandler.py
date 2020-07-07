from models.projectors.NickelProjector import NickelProjector 
from models.projectors.Type2VecProjector import Type2VecProjector
from torch.nn import Module, ModuleList

class MultiProjectorHandler(Module):

  def __init__(self, config):

    super().__init__()
    
    self.nametag = 'MultiProjectorsHandler'

    self.setup_classes_factory()

    self.names = config[self.nametag]['PROJECTOR_CONFIGS'].split(' ')

    self.classes = self.get_classes(config)

    self.projectors = ModuleList([cl(config) for cl in self.classes])
    print('projectors: {}'.format(self.projectors))

  def setup_classes_factory(self):
    self.factory_dict = {
      "HyperbolicProjector": NickelProjector,
      'CosineProjector': Type2VecProjector
    }

  def get_classes(self, conf):
    classes = []
    for name in self.names:
      classes.append(self.factory_dict[conf[name]['Class']])
    return classes
  
  def forward(self, vec):
    projections = []
    for projector in self.projectors:
      projections.append(projector(vec))
    return projections




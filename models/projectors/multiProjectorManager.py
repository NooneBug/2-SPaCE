from models.projectors.NickelProjector import NickelProjector 
from models.projectors.Type2VecProjector import Type2VecProjector
from torch.nn import Module, ModuleDict

class MultiProjectorManager(Module):

  def __init__(self, config):

    super().__init__()
    
    self.nametag = 'MultiProjectorsManager'

    self.setup_classes_factory()
		
    self.classes = self.get_classes(config)

    self.projectors = ModuleDict({k:cl(config) for k, cl in self.classes.items()})
    # print('projectors: {}'.format(self.projectors))

  def setup_classes_factory(self):
    self.factory_dict = {
      "HyperbolicProjector": NickelProjector,
      'CosineProjector': Type2VecProjector
    }

  def get_classes(self, conf):
    self.names = conf[self.nametag]['PROJECTOR_CONFIGS'].split(' ')
    classes = {}
    for name in self.names:
      classes[conf[name]['NAME']] = self.factory_dict[conf[name]['Class']]
    return classes
  
  def forward(self, vec):
    projections = {}
    for k, projector in self.projectors.items():
      projections[k] = projector(vec)
    return projections




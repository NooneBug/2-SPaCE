

from models.projectors.NickelProjector import NickelProjector 



class MultiProjectorHandler():

	def __init__(self, config):
		
		self.setup_classes_factory()
	
	def setup_classes_factory(self):
		self.factory_dict = {
      "HyperbolicProjector": NickelProjector
		}

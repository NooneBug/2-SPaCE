from losses.losses import CosineDissimilarityLoss

class MultiLossHandler():
	
	def __init__(self, config):
		self.nametag = 'MultiLossHandler'
		self.conf = config[self.nametag]

	def define_factory():
		self.loss_factory = {
			'cosine_dissimilarity': CosineDissimilarityLoss,
			# 'hyperbolic_distance': HyperbolicDistanceLoss,
			'normalized_hyperbolic_distance': 'NHYPD',
			'regularized_hyperbolic_distance': 'RHYPD',
			'hyperboloid_distance' : 'LORENTZD',
			'multilabel_Minimum_Normalized_Poincare': 'NHMML',
			'multilabel_Minimum_Poincare': 'HMML',
			'multilabel_Minimum_Cosine': 'DMML',
			'multilabel_Average_Poincare': 'HMAL',
			'multilabel_Average_Cosine': 'DMAL', 
		}

from abc import ABC

class Loss(ABC):
	def compute_loss(self, actual_vectors, true_vectors):
		pass

class CosineDissimilarityLoss(Loss):
	def compute_loss(self, actual_vectors, true_vectors):
		pass

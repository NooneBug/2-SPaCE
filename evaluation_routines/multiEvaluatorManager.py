

class MultiEvaluatorManager():

	def __init__(self, model, testLoader, config):
		self.nametag = 'MultiEvaluatorManager'

		self.trained_model = model

		self.testLoader = testLoader

		self.config = config

		self.initialize_evaluators()

	def initialize_evaluators(self):
		

	# def initialize_factory(self):
	# 	self.evaluator_factory = {
	# 		'classification_evaluator': ClassificationEvaluator,
	# 		'regression_evaluator': RegressionEvaluator,
	# 	}


	def evaluate():
			pass
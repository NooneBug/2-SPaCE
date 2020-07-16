from geoopt.optim import RiemannianAdam


class RiemannianAdamOptimizer(RiemannianAdam):

	def __init__(self, config, model):
		nametag = 'RIEMANNIAN_ADAM'

		learning_rate = float(config[nametag]['learning_rate'])

		return super().__init__(model.parameters(), lr= learning_rate)

from torch.nn import Module
from torch import nn

class CommonNetwork(Module):

	def __init__(self, config):
		super().__init__()
		self.nametag = 'COMMON'
		self.conf = dict(config[self.nametag])

		self.cast_params()
        
		prec = self.input_dim
		self.fully = nn.ModuleList()
		self.bns = nn.ModuleList()

		for dim in self.layers:
				self.fully.append(nn.Linear(prec, dim).cuda())
				self.bns.append(nn.BatchNorm1d(dim).cuda())
				prec = dim            
		
		self.dropout = nn.Dropout(p=self.dropout_prob).cuda()
		self.leaky_relu = nn.LeakyReLU(0.1).cuda()

		for layer in self.fully:
				nn.init.xavier_normal_(layer.weight)
				if layer.bias is not None:
						nn.init.zeros_(layer.bias)

	def cast_params(self):
		self.layers = [int(splitted) for splitted in self.conf['layers'].split(' ')]
		self.input_dim = int(self.conf['input_size'])
		self.dropout_prob = float(self.conf['dropout_prob'])

	def forward(self, x):
		for i in range(len(self.fully)):
			# x = x.double()
			x = self.dropout(self.bns[i](self.leaky_relu(self.fully[i](x))))
		return x

from embedding_managers.embedding_interface import Embedding
from embedding_managers.nickel_embedding import NickelEmbedding
from embedding_managers.type2vec_embedding import Type2VecEmbeddingManager

import torch


class MultiEmbeddingManager(Embedding):

	def __init__(self, spaces_number, names, classes):	
		self.embeddings = {n: {} for n in names}
		self.n = spaces_number
		
		self.setup_classes_factory()

		self.check_names(var = classes)
		self.classes = classes

	def setup_classes_factory(self):
		self.factory_dict = {
			"NickelEmbedding" : NickelEmbedding,
			"Type2VecEmbedding": Type2VecEmbeddingManager
		}

	def check_names(self, var):
		if not all(elem in list(self.embeddings.keys()) for elem in var):
			raise Exception('some names in the dict does not match with the names of embeddings\n \
				embedding names: {}; given names: {}'.format(list(self.embeddings.keys()), list(var.keys())))


	def load_from_file(self, paths):
		''' variable "paths" is a dict with previous embedding names and paths '''
		if type(paths) != dict:
			raise Exception('variable paths needs to be a dict')
		
		self.check_names(var = paths)
		
		names = list(self.embeddings.keys())
		print('-----------------------------------------------------------------------------------')
		print(f" Retrieving the following configuration(s): {', '.join(names[:-1])} and {names[-1]}")
		print('-----------------------------------------------------------------------------------')

		for name in names:
			# initialize with factory
			self.embeddings[name] = self.factory_dict[self.classes[name]]()
			# load embeddings
			self.embeddings[name].load_from_file(paths[name])
			
		self.torchify(names)

		self.generate_token2idx_dict()

		return self.embeddings

	def torchify(self, names):
		for name in names:
			e = self.embeddings[name].embeddings 
			self.embeddings[name].embeddings = {k: torch.tensor(v) if not torch.is_tensor(v) else v for k, v in e.items()}
	
	def generate_token2idx_dict(self):
		self.token2idx_dict = {}
		tokens = set()
		for k, v in self.embeddings.items():
			tokens = tokens.union(v.embeddings.keys())
		
		for i, t in enumerate(tokens):
			self.token2idx_dict[t] = i

	def token2idx(self, token):
		return self.token2idx_dict[token]

	def generate_lookup_network(self):
		return {name: model.generate_lookup_network() for name, model in self.embeddings.items()}
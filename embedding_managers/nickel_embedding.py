from embedding_managers.embedding_interface import Embedding
import torch


class NickelEmbedding(Embedding):
	
	def load_from_file(self, path):
		self.embeddings = torch.load(path)
		
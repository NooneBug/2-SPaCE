from embedding_managers.embedding_interface import Embedding
from common.utils import load_data_with_pickle

class Type2VecEmbeddingManager(Embedding):
	def load_from_file(self, path):
		self.embeddings = load_data_with_pickle(path)
from embedding_managers.embedding_interface import Embedding
from common.utils import load_data_with_pickle
from models.lookup_models.lookup_networks import LookupNetwork

class Type2VecEmbeddingManager(Embedding):
  def load_from_file(self, path):
    self.embeddings = load_data_with_pickle(path)

  def get_vec(self, label):
    if label in self.embeddings:
      return self.token2vec[label]
    else:
      raise Exception('label {} not present in embeddings'.format(label))

  def get_embeddings_number(self):
    return len(self.embeddings)

  def get_vector_dim(self):
    return len(list(self.embeddings.values())[0])

  def generate_lookup_network(self):
    return LookupNetwork(self)
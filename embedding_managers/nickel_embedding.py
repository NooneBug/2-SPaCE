from embedding_managers.embedding_interface import Embedding
import torch
from models.lookup_networks import LookupNetwork


class NickelEmbedding(Embedding):
  
  def load_from_file(self, path):
    self.embeddings = torch.load(path)

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
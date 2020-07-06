from embedding_managers.embedding_interface import Embedding
import torch
from models.lookup_models.lookup_networks import LookupNetwork


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
    lookup_network = LookupNetwork(self)
    
    # weights = torch.tensor([self.id2vec(id) for id in range(self.get_embeddings_number())])

    # lookup_net.weight.data.copy_(weights)
    
    return lookup_network

  # def id2vec(self, id):
  #     return self.get_vec(self.idx2token(id))
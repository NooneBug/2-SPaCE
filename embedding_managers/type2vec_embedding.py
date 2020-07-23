from embedding_managers.embedding_interface import Embedding
from common.utils import load_data_with_pickle
from models.lookup_models.lookup_networks import LookupNetwork
import torch

class Type2VecEmbeddingManager(Embedding):
  def load_from_file(self, path):
    self.embeddings = load_data_with_pickle(path)

  def get_vec(self, label):
    # print('label: {}'.format(label))
    # print('embeddings: {}'.format(self.embeddings))
    if label in self.embeddings:
      return self.embeddings[label]
    else:
      # return torch.rand(self.get_vector_dim())
      raise Exception('label {} not present in embeddings'.format(label))
  
  def get_embeddings_number(self):
    return len(self.embeddings)

  def get_vector_dim(self):
    return len(list(self.embeddings.values())[0])

  def generate_lookup_network(self, padding_idx):
    
    # while len(self.embeddings) != len(self.token2idx_dict):
      # self.embeddings = torch.cat((self.embeddings, torch.rand(self.get_vector_dim())), 0)

    lookup_network = LookupNetwork(self, padding_idx=padding_idx)
    
    weights = torch.tensor([self.idx2vec(idx).numpy() for idx in range(self.get_embeddings_number())])

    lookup_network.model.weight.data.copy_(weights)
    lookup_network.model.weight.requires_grad = False

    
    return lookup_network

  def idx2vec(self, idx):
    return self.get_vec(self.idx2token(idx))

  def set_token2idx_dict(self, token2idx_dict):
    self.token2idx_dict = token2idx_dict
    self.idx2token_dict = {v:k for k, v in token2idx_dict.items()}

  def idx2token(self, id):
    return self.idx2token_dict[id]

  def get_ordered_typelist(self):
    return list(self.embeddings.keys())

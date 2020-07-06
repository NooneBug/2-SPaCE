from embedding_managers.embedding_interface import Embedding
from tqdm import tqdm
import torch
from models.lookup_models.lookup_networks import LookupNetwork

class glove_word_embedding(Embedding):

  def __init__(self):
    self.token2vec = {}
    self.token2idx_dict = {}

  def generate_lookup_network(self, padding_idx):
    if padding_idx:
      lookup_net = LookupNetwork(self, padding_idx=padding_idx)
    else:
      lookup_net = LookupNetwork(self)


    weights = [self.idx2vec(id).numpy() for id in range(self.get_embeddings_number())]

    weights = torch.tensor(weights)

    lookup_net.model.weight.data.copy_(weights)

    return lookup_net

  def get_embeddings_number(self):
    return len(self.token2vec)

  def get_vector_dim(self):
    return len(list(self.token2vec.values())[0])

  def load_from_file(self, path):
    print("Start loading pretrained word vecs")
    for line in tqdm(open(path)):
      fields = line.strip().split()
      token = fields[0]
      try:
        vec = list(map(float, fields[1:]))
      except ValueError:
        continue
      self.add(token, torch.Tensor(vec))
    self.create_id2token()
  
  def add(self, token, vector):
    self.token2vec[token] = vector
    if token not in self.token2idx_dict:
      try:
        self.token2idx_dict[token] = max(self.token2idx_dict.values()) + 1
      except:
        self.token2idx_dict[token] = 0
  
  def get_vec(self, word):
    if word in self.token2vec:
      return self.token2vec[word]
    else:
      raise Exception('Word {} not present in embeddings'.format(word))

  def __contains__(self, word):
    return word in self.token2vec 

  def create_id2token(self):
    self.id2token_dict = {v:k for k, v in self.token2idx_dict.items()}

  def token2idx(self, token):
    return self.token2idx_dict[token]

  def idx2token(self, id):
    return self.id2token_dict[id]

  def idx2vec(self, id):
    return self.token2vec[self.idx2token(id)]


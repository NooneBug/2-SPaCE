from embedding_managers.embedding_interface import Embedding
from tqdm import tqdm
import torch
from models.lookup_models.lookup_networks import LookupNetwork

class glove_word_embedding(Embedding):

  def __init__(self):
    self.token2vec = {}
    self.token2idx_dict = {}
    self.unknown_token = 'UNK'
    self.idx_counter = 0

  def generate_lookup_network(self, padding_idx = None):
    if type(padding_idx) == int:
      lookup_net = LookupNetwork(self, padding_idx=padding_idx)
    else:
      lookup_net = LookupNetwork(self)

    self.padding_idx = padding_idx

    
    weights = [self.idx2vec(idx).numpy() for idx in range(self.get_embeddings_number())]

    weights = torch.tensor(weights)

    lookup_net.model.weight.data.copy_(weights)
    lookup_net.model.weight.requires_grad = False

    return lookup_net

  def get_embeddings_number(self):
    return len(self.token2vec)

  def get_vector_dim(self):
    return len(list(self.token2vec.values())[0])

  def wc(self, files):
    if not isinstance(files, list) and not isinstance(files, tuple):
        files = [files]
    return sum([sum([1 for _ in open(fp, buffering=64 * (1024 ** 2))]) for fp in files])

  def load_from_file(self, path):
    print("Start loading pretrained word vecs")
    for line in tqdm(open(path), total=self.wc(path)):
      fields = line.strip().split()
      token = fields[0]
      try:
        vec = list(map(float, fields[1:]))
      except ValueError:
        continue
      self.add(token, torch.Tensor(vec))

    self.add(self.unknown_token, torch.zeros(size= (len(vec),)))
    
    self.create_id2token()

  
  def add(self, token, vector):
    # if token not in self.token_set:
    # if token not in self.token2idx_dict:
    if token not in self.token2vec:
      self.token2vec[token] = vector
      self.token2idx_dict[token] = self.idx_counter
      self.idx_counter += 1
    # try:
    #   self.token2idx_dict[token] = max(self.token2idx_dict.values()) + 1
    # except:
    #   self.token2idx_dict[token] = 0
  
  def get_vec(self, word):
    if word in self.token2vec:
      return self.token2vec[word]
    else:
      return self.token2vec[self.unknown_token]
      # raise Exception('Word {} not present in embeddings'.format(word))

  def __contains__(self, word):
    return word in self.token2vec 

  def create_id2token(self):
    self.id2token_dict = {v:k for k, v in self.token2idx_dict.items()}
    
    mancanti = []
    for i in range(self.get_embeddings_number()):
      try:
        a = self.id2token_dict[i]
      except:
        mancanti.append(i)
    print('mancano: {}'.format(mancanti))

  def token2idx(self, token):
    if token in self.token2vec:
      return self.token2idx_dict[token]
    else:
      return self.token2idx_dict[self.unknown_token]

  def idx2token(self, idx):
    if idx != self.padding_idx:
      return self.id2token_dict[idx]
    
  def idx2vec(self, idx):
    return self.token2vec[self.idx2token(idx)]


from embedding_managers.embedding_interface import Embedding
from tqdm import tqdm
import torch

class glove_word_embedding(Embedding):

  def __init__(self):
    self.token2vec = {}
    self.token2idx_dict = {}

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


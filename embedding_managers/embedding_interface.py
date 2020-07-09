from abc import ABC
from models.lookup_models.lookup_networks import LookupNetwork

class Embedding(ABC):

  def load_from_file(self, path: str) -> dict:
    """load the embeddings and return a dictionary with key:index"""
    pass

  def token2idx(self, token:str) -> int:
    """return the index of the input token"""
    pass

  def idx2token(self, index:int) -> str:
    """return the token correspondant to the input index"""
    pass

  def get_embeddings_number(self) -> int:
    """return the number of vectors in this embedding"""
    pass
  
  def get_vector_dim(self) -> int:
    """return the dimension of vectors in this embedding"""
    pass

  def generate_lookup_network(self, padding_index:int) -> LookupNetwork:
    """" generate and returns a lookup network on the embeddings"""
    pass
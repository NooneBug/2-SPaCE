from abc import ABC


class Embedding(ABC):

  def load_from_file(self, path: str) -> dict:
    "load the embeddings and return a dictionary with key:index"
    pass

  def token2idx(self, token:str) -> int:
    "return the index of the input token"
    pass

  def idx2token(self, index:int) -> str:
    "return the token correspondant to the input index"
    pass
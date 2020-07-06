from torch.nn import Embedding



class LookupNetwork:

	def __init__(self, embedding, padding_idx = None):
		if padding_idx:
			self.model = Embedding(num_embeddings = embedding.get_embeddings_number(), 
															embedding_dim= embedding.get_vector_dim(),
															padding_idx=padding_idx)
		else:
			self.model = Embedding(num_embeddings = embedding.get_embeddings_number(), 
															embedding_dim= embedding.get_vector_dim())
			

	
	
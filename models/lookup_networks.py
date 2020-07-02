from torch.nn import Embedding



class LookupNetwork:

	def __init__(self, embedding):
		model = Embedding(num_embeddings = embedding.get_embeddings_number(), 
											embedding_dim= embedding.get_vector_dim())
	
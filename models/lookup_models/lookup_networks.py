from torch.nn import Embedding, Module
import torch

class LookupNetwork(Module):

	def __init__(self, embedding, padding_idx = None):
		super().__init__()
		if type(padding_idx) == int:
			self.model = Embedding(num_embeddings = embedding.get_embeddings_number(), 
															embedding_dim = embedding.get_vector_dim(),
															padding_idx = padding_idx).cuda()
		else:
			self.model = Embedding(num_embeddings = embedding.get_embeddings_number(), 
															embedding_dim = embedding.get_vector_dim()).cuda()
		self.emb = embedding
		self.padding_idx = padding_idx
		self.embedding_dim = embedding.get_vector_dim()
		self.embedding_number = embedding.get_embeddings_number()

	def forward(self, input_batch):
		batch = []
		for i, entry in enumerate(input_batch):
			ret = []
			for j, idx in enumerate(entry):
				if idx == self.padding_idx:
					ret.append(torch.zeros(size = (self.embedding_dim,)))
				else:
					idx = idx.clone().cuda()
					ret.append(self.model(idx).detach().cpu().numpy())
			batch.append(ret)
		return torch.tensor(batch).cuda()
			
	
from torch.nn import Embedding, Module
import torch

class LookupNetwork(Module):

	def __init__(self, embedding, padding_idx = None, no_label_idx = -1):
		super().__init__()
		if type(padding_idx) == int:
			self.model = Embedding(num_embeddings = embedding.get_embeddings_number(), 
															embedding_dim = embedding.get_vector_dim(),
															padding_idx = padding_idx).cuda()
		else:
			self.model = Embedding(num_embeddings = embedding.get_embeddings_number(), 
															embedding_dim = embedding.get_vector_dim()).cuda()
		self.emb = embedding
		self.no_label_idx = no_label_idx
		self.embedding_dim = embedding.get_vector_dim()
		self.embedding_number = embedding.get_embeddings_number()

	def forward(self, input_batch):
		batch = []
		for i, entry in enumerate(input_batch):
			ret = []
			for j, idx in enumerate(entry):
				if idx.item() == self.no_label_idx:
					ret.append(torch.zeros(size = (self.embedding_dim,)).cuda())
				else:
					# idx = idx.clone().cuda()
					# print('idx : {}'.format(idx))
					ret.append(self.model(idx))
			batch.append(torch.stack(ret))
		return torch.stack(batch).cuda()
		
	# def forward(self, input_batch):
	# 	return self.model(input_batch)
	
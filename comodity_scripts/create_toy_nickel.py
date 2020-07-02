import torch

if __name__ == "__main__":


	emb = torch.load('resources/type_embeddings/ontonotes_nickel.pth')

	types = ['animal', 
						'location', 'city']

	extracted_emb = {}

	for t in types:
		extracted_emb[t] = emb[t]

	torch.save(extracted_emb, 'resources/type_embeddings/toy_nickel.pth')	
	
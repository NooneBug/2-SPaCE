import pickle
from common.util import load_data_with_pickle, save_data_with_pickle

def main():
	emb = load_data_with_pickle('resources/type_embeddings/type2vec')

	types = ['animal', 
						'location', 'city']

	extracted_emb = {}

	for t in types:
		extracted_emb[t] = emb[t]

	save_data_with_pickle('resources/type_embeddings/toy_t2v', extracted_emb)	
	
if __name__ == "__main__":
	main()
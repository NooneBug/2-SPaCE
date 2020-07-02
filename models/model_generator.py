from parsers.dataset_parser import get_parsed_datasets
from models.lookup_models.lookup_networks import LookupNetwork
from models.toSPaCE import ComposedNetwork

def main(config):

	dataset, word_embedding, type_embedding, encoded_dataset = get_parsed_datasets(config)

	word_embedding_lookup = word_embedding.generate_lookup_network()

	type_embedding_lookup = type_embedding.generate_lookup_network()

	model = ComposedNetwork(config = config,
													word_lookup = word_embedding_lookup,
													type_lookup = type_embedding_lookup,
													)

	print(model.word_network)

if __name__ == "__main__":
	main()

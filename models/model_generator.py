from parsers.dataset_parser import get_parsed_datasets
from models.lookup_networks import LookupNetwork

def main():
	dataset, word_embedding, type_embedding, encoded_dataset = get_parsed_datasets()

	word_embedding_lookup = word_embedding.generate_lookup_network()

	type_embedding_lookup = type_embedding.generate_lookup_network()

	print(word_embedding_lookup)

	print('-----------------')

	print(type_embedding_lookup)


if __name__ == "__main__":
	main()

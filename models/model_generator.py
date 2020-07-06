from models.toSPaCE import ComposedNetwork

def generate_composed_model(word_embedding_lookup, type_embedding_lookup, config):

	model = ComposedNetwork(config = config,
													word_lookup = word_embedding_lookup,
													type_lookup = type_embedding_lookup,
													)


	return model


if __name__ == "__main__":
	generate_composed_model()

from models.toSPaCE import ComposedRegressiveNetwork, ComposedClassificationNetwork



def generate_composed_model(word_embedding_lookup, type_embedding_lookup, config):

	network_class = config['2-SPACE MODULES CONFIGS']['CLASS']

	factory = get_factory()
	model = factory[network_class](config = config,
																	word_lookup = word_embedding_lookup,
																	type_lookup = type_embedding_lookup,
																	)


	return model

def get_factory():
	return {
		'ComposedRegressiveNetwork': ComposedRegressiveNetwork,
		'ComposedClassificationNetwork': ComposedClassificationNetwork
	}


if __name__ == "__main__":
	generate_composed_model()

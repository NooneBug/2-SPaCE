from models.model_generator import generate_composed_model
from parsers.dataset_parser import get_parsed_datasets
from models.lookup_models.lookup_networks import LookupNetwork

TRAIN_CONFIG = 'TRAINING_PARAMETERS'


def train(config):
	trainLoader, valLoader, model = training_setup(config)

	# print(model)

	training_routine(trainLoader, valLoader, model)

def training_routine(train_loader, val_loader, model):
	model.train_(train_loader, val_loader)

def training_setup(config):
	configuration_dataset, word_embedding, type_embedding, val_dataset = get_parsed_datasets(config)

	if config.has_option(section = config['2-SPACE MODULES CONFIGS']['WORD_MANIPULATION_MODULE'], 
												option = 'PADDING_INDEX'):
		pad = int(config[config['2-SPACE MODULES CONFIGS']['WORD_MANIPULATION_MODULE']]['PADDING_INDEX'])
		word_embedding_lookup = word_embedding.generate_lookup_network(padding_idx = pad)
	else:
		word_embedding_lookup = word_embedding.generate_lookup_network()

	type_embedding_lookup = type_embedding.generate_lookup_network()

	model = generate_composed_model(word_embedding_lookup=word_embedding_lookup,
																	type_embedding_lookup=type_embedding_lookup,
																	config = config)

	model.set_optimizer(config)

	train_config = config[TRAIN_CONFIG]

	# print('--------------------')
	# print('configuration_dataset: {}'.format(configuration_dataset))

	trainLoader = model.get_DataLoader(dataset = configuration_dataset, 
																			batch_size = int(train_config['train_batch_size']), 
																			shuffle = bool(train_config['shuffle']))

	valLoader = model.get_DataLoader(dataset = val_dataset, 
																		batch_size= int(train_config['val_batch_size']))

	# for i, data in enumerate(trainLoader):
		# print('-----------------------')
		# print('{}: {}'.format(i, data))

	return trainLoader, valLoader, model

if __name__ == "__main__":
		train()
from models.model_generator import generate_composed_model
from parsers.dataset_parser import get_parsed_datasets
from models.lookup_models.lookup_networks import LookupNetwork
from datetime import date
import os

def get_today():
	return date.today().strftime('%m_%d_%Y')


TRAIN_CONFIG = 'TRAINING_PARAMETERS'

def setup_result_folder(config):
	if config['DEFAULT']['RESULT_FILE'] == 'auto':
		foldername = config['DEFAULT']['RESULT_FOLDER'] + get_today()
	else:
		foldername = config['DEFAULT']['RESULT_FOLDER'] + config['DEFAULT']['RESULT_FILE']
	can_exit = False
	foldername = foldername + '_0'
	i = 0
	while not can_exit:
		if not os.path.exists(foldername):
			os.mkdir(foldername)
			can_exit = True
		else:
			foldername = '_'.join(foldername.split('_')[:3])
			i += 1
			foldername = foldername + '_{}'.format(i)
		
	return foldername

def initialize_folder(foldername, config):
	with open(foldername + '/config.ini', 'w') as out:
		config.write(out)

def evaluate(trained_model, testLoader, config):
	foldername = setup_result_folder(config)
	initialize_folder(foldername, config)
	trained_model.evaluate(testLoader, foldername)

def train(trainLoader, valLoader, model, config):
	# print(model)

	return training_routine(trainLoader, valLoader, model)

def training_routine(train_loader, val_loader, model):
	return model.train_(train_loader, val_loader)

def get_datas_and_model(config):
	configuration_dataset, word_embedding, type_embedding, val_dataset, test_dataset = get_parsed_datasets(config)

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
	
	testLoader = model.get_DataLoader(dataset=test_dataset, 
																		batch_size= int(train_config['test_batch_size']))

	# for i, data in enumerate(trainLoader):
		# print('-----------------------')
		# print('{}: {}'.format(i, data))

	return trainLoader, valLoader, testLoader, model


def routine(config):
	trainLoader, valLoader, testLoader, model = get_datas_and_model(config)

	trained_model = train(trainLoader, valLoader, model, config)

	evaluate(trained_model, testLoader, config)
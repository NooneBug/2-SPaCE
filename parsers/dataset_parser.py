import configparser
import json
from embedding_managers.glove_word_embedding import glove_word_embedding
from embedding_managers.multi_type_embedding_manager import MultiEmbeddingManager

WORD_EMBEDDING_CONFIG = "GLOVE"
TYPE_EMBEDDING_CONFIG = "MultiTypeEmbedding"

classes_dict = {
  'GLOVE' : glove_word_embedding,
  'MultiTypeEmbedding': MultiEmbeddingManager
}

def get_parsed_datasets(config):

  dataset = parse_dataset(path = config[WORD_EMBEDDING_CONFIG]["DATASET_PATH"])

  word_embeddings, type_embeddings = obtain_embeddings(config=config)

  encoded_datasets = get_encoded_dataset(dataset, word_embeddings, type_embeddings, config)

  return dataset, word_embeddings, type_embeddings, encoded_datasets


def get_encoded_dataset(dataset, word_embeddings, type_embeddings, config):
  config_default_dict = config['DEFAULT']
  encoded_dataset = []
  # print(word_embeddings.token2idx_dict)
  for d in dataset:
    encoded_entry = {}
    # print('------------------------------')
    # print('original entry: {}'.format(d))
    fields = [config_default_dict[value] for value in ['LEFT_CONTEXT', 'RIGHT_CONTEXT', 'MENTION']]
    for token_list, field in [(d[f], f) for f in fields]:
      if type(token_list) != list:
        token_list = token_list.split(' ')
      # print(token_list)
      encoded_entry[field] = [word_embeddings.token2idx(token) for token in token_list]


    labels_fields = [config_default_dict[v] for v in ['LABELS']]

    for labels, field in [(d[f], f) for f in labels_fields]:
      # print(labels)

      encoded_entry[field] = [type_embeddings.token2idx(l) for l in labels]

    # print('encoded_entry: {}'.format(encoded_entry))

    encoded_dataset.append(encoded_entry)
  # print(encoded_dataset)

  return encoded_dataset
def obtain_embeddings(config):
  print('-----------------------------------------------------------------------------------')
  print(' Loading word embeddings')
  print('-----------------------------------------------------------------------------------')
  
  word_embeddings = classes_dict[WORD_EMBEDDING_CONFIG]()

  word_embeddings.load_from_file(config[WORD_EMBEDDING_CONFIG]["WORD_EMBEDDING_PATH"])
  
  if int(config['DEFAULT']['TYPE_SPACE_NUMBER']) >= 1:
    if TYPE_EMBEDDING_CONFIG == "MultiTypeEmbedding":
      type_embeddings = retrieve_multi_type_embeddings(config)
      
    else:
      raise Exception('you modified the TYPE_EMBEDDING_CONFIG name without take care of its usage; \n please, check the type_embeddings loading routine...')
  else:
    raise Exception('please, setup a routine for single-output configurations (now those are managed by MultiTypeEmbedding class')

  # for k, v in type_embeddings.items():
  #   print('-------------------------------------')
  #   print('{}:\n'.format(k))
  #   for k2, v2, in v.embeddings.items():
  #     print('\t{}:\t{}\n'.format(k2, v2))

  return word_embeddings, type_embeddings

def retrieve_multi_type_embeddings(config):

  embedding_config_names = config[TYPE_EMBEDDING_CONFIG]["EMBEDDING_CONFIGS"].split(' ')

  if len(embedding_config_names) != int(config['DEFAULT']['TYPE_SPACE_NUMBER']):
    raise Exception('ERROR: the number of type space(s) and their names/configuration in config.ini does not match')

  names = [config[n]["NAME"] for n in embedding_config_names]
  paths = [config[n]["PATH"] for n in embedding_config_names]
  classes = [config[n]["EMBEDDING_CLASS_NAME"] for n in embedding_config_names]

  type_embeddings = classes_dict[TYPE_EMBEDDING_CONFIG](spaces_number=len(names), 
                                                        names=names,
                                                        classes = get_classes_dict(names, classes))
  type_embeddings.load_from_file(get_paths_dict(names, paths))

  return type_embeddings


def get_classes_dict(names, classes):
  return {n: v for n, v in zip(names, classes)}

def get_paths_dict(names, paths):
  return {n: v for n, v in zip(names, paths)}  


def parse_dataset(path):
  with open(path, 'r') as inp:
    d = json.load(inp)
  return d['datas']
  
if __name__ == "__main__":
  main()
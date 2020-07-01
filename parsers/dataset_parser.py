import configparser
import json
from embedding_managers.glove_word_embedding import glove_word_embedding
from embedding_managers.multi_type_embedding_manager import MultiEmbeddingManager

WORD_EMBEDDING_CONFIG = "GLOVE"
TYPE_EMBEDDING_CONFIG = "MultiTypeEmbedding"

def main():
  config = configparser.ConfigParser()

  config.read("parsers/config.ini")


  dataset = parse_dataset(path = config[WORD_EMBEDDING_CONFIG]["DATASET_PATH"])

  word_embeddings = glove_word_embedding()

  word_embeddings.load_from_file(config[WORD_EMBEDDING_CONFIG]["WORD_EMBEDDING_PATH"])

  if TYPE_EMBEDDING_CONFIG == "MultiTypeEmbedding":

    type_embeddings = retrieve_multi_type_embeddings(config)
    
  else:
    raise Exception('you modified the TYPE_EMBEDDING_CONFIG name without take care of its usage; \n please, check the type_embeddings loading routine...')


  print(type_embeddings)



def retrieve_multi_type_embeddings(config):

  embedding_config_names = config[TYPE_EMBEDDING_CONFIG]["EMBEDDING_CONFIGS"].split(' ')

  names = [config[n]["NAME"] for n in embedding_config_names]
  paths = [config[n]["PATH"] for n in embedding_config_names]
  classes = [config[n]["EMBEDDING_CLASS_NAME"] for n in embedding_config_names]

  type_embeddings = MultiEmbeddingManager(spaces_number=len(names), 
                                          names=names,
                                          classes = get_classes_dict(names, classes))
  type_embeddings.load_from_files(get_paths_dict(names, paths))

  return type_embeddings.embeddings


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
import configparser

# from parsers.dataset_parser import get_parsed_datasets as main
# from models.model_generator import main
# from comodity_scripts.create_toy_type2vec import main
from training_routines.toy_training import routine as main


if __name__ == "__main__":

  config = configparser.ConfigParser()

  # config.read("toy_config.ini")
  config.read("config.ini")

  main(config)
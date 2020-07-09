from parsers.parser_interface import Parser
from models.input_encoder_models.ShimaokaModel import CharEncoder
import torch


class ShimaokaParser(Parser):

  def __init__(self, config):
    self.nametag = 'SHIMAOKA'
    self.conf = dict(config[self.nametag])
    self.cast_params()

  def cast_params(self):
    self.cast_param('positional_emb_size', int)
    self.cast_param('mention_length', int)
    self.cast_param('max_char_in_mention', int)

    self.cast_param('context_rnn_size', int)
    self.cast_param('emb_size', int)
    self.cast_param('char_emb_size', int)
    self.cast_param('positional_emb_size', int)
    self.cast_param('mention_dropout_size', float)
    self.cast_param('context_dropout', float)

    self.cast_param('padding_index', int)

  def cast_param(self, key, cast_type):
    self.conf[key] = cast_type(self.conf[key])
  
  def cast_dataset(self, dataset, encoded_dataset, config):

    config_default_dict = config['DEFAULT']

    sentence_max_length = self.conf['positional_emb_size']
    mention_max_length =  self.conf['mention_length']

    configuration_dataset = []

    for entry, encoded_entry in zip(dataset, encoded_dataset):
      configuration_entry = []

      left_context = encoded_entry[config_default_dict['LEFT_CONTEXT']]
      right_context = encoded_entry[config_default_dict['RIGHT_CONTEXT']]
      mention = encoded_entry[config_default_dict['MENTION']]
      labels = encoded_entry[config_default_dict['LABELS']]

      encoded_sentence = encoded_entry[config_default_dict['LEFT_CONTEXT']] + encoded_entry[config_default_dict['MENTION']] + encoded_entry[config_default_dict['RIGHT_CONTEXT']] 

      sentence_length = torch.tensor([len(encoded_sentence)]).cuda()
      
      while len(encoded_sentence) < sentence_max_length:
        encoded_sentence.append(self.conf['padding_index'])

      context_positions = [-1 for i in range(sentence_max_length)]

      for i in range(sentence_max_length):
        if i < len(left_context):
          context_positions[i] = float(- (len(left_context) - i))
        elif i < len(left_context) + len(mention):
          context_positions[i] = float(0)
        elif i < len(encoded_sentence):
          context_positions[i] = float(i - (len(left_context) + len(mention)) + 1)
    
      encoded_mention = self.get_encoded_mention(encoded_entry[config_default_dict['MENTION']], 
                                                  self.conf['mention_length'])

      encoded_mention_chars = self.encode_chars(entry[config_default_dict['MENTION']], 
                                                self.conf['max_char_in_mention'])

      encoded_labels = encoded_entry[config_default_dict['LABELS']]

      configuration_entry = [encoded_sentence, context_positions, sentence_length, 
                              encoded_mention, encoded_mention_chars, encoded_labels]
      
      configuration_dataset.append(configuration_entry)
      print('------------------------')
      print('entry: {}'.format(entry))
      print('encoded_entry: {}'.format(encoded_entry))
      print('encoded_sentence: {}'.format(encoded_sentence))
      print('context_positions: {}'.format(context_positions))
      print('sentence_length: {}'.format([sentence_length]))
      print('encoded_mention: {}'.format(encoded_mention))
      print('encoded_mention_chars: {}'.format(encoded_mention_chars))
      print('encoded_labels: {}'.format(encoded_labels))

    return configuration_dataset

  def get_encoded_mention(self, mention, mention_lenght):
    encoded_mention = [0 for i in range(mention_lenght)]

    for i, elem in enumerate(mention):
      if i < mention_lenght:
        encoded_mention[i] = elem
    return encoded_mention    

  def encode_chars(self, mention, max_length):
    
    char_encoder = CharEncoder(self.conf)

    encoded_mention_chars = []
    mention = mention[:max_length]
    for char in mention:
      encoded_mention_chars.append(char_encoder.CHARS.index(char))
    while len(encoded_mention_chars) < max_length:
      encoded_mention_chars.append(char_encoder.CHARS.index(self.conf['char_pad']))
    return encoded_mention_chars
      

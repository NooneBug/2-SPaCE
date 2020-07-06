from torch.utils.data import Dataset
import torch

class ShimaokaDataset(Dataset):
  def __init__(self, shimaoka_parsed_input):
    self.input = shimaoka_parsed_input
    self.encoded_sentences = torch.tensor(self.extract_data(0))
    self.context_positions = torch.tensor(self.extract_data(1))
    self.sentence_length = torch.tensor(self.extract_data(2))
    self.encoded_mention = torch.tensor(self.extract_data(3))
    self.encoded_mention_chars = torch.tensor(self.extract_data(4))
    self.encoded_labels = torch.tensor(self.extract_data(5))

  def __getitem__(self, index):
    return (self.encoded_sentences[index], self.context_positions[index], 
            self.sentence_length[index], self.encoded_mention[index], 
            self.encoded_mention_chars[index], self.encoded_labels[index])
  
  def __len__(self):
        return len(self.input)

  def extract_data(self, index):
    return [d[index] for d in self.input]
  
    

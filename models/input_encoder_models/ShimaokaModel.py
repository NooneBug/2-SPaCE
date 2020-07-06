from torch import nn
from models.lookup_models import lookup_networks
from torch.nn import Module

class ShimaokaMentionAndContextEncoder(Module):

  def __init__(self, config):
    super().__init__()
    
    self.nametag = 'SHIMAOKA'
    self.conf = dict(config[self.nametag])
    self.cast_params()
    
    self.mention_encoder = MentionEncoder(self.conf).cuda()
    self.context_encoder = ContextEncoder(self.conf).cuda()
    self.feature_len = self.conf['context_rnn_size'] * 2 + self.conf['emb_size'] + self.conf['char_emb_size']

  def cast_params(self):
    self.cast_param('context_rnn_size', int)
    self.cast_param('emb_size', int)
    self.cast_param('char_emb_size', int)
    self.cast_param('positional_emb_size', int)
    self.cast_param('mention_dropout_size', float)
    self.cast_param('context_dropout', float)

  def cast_param(self, key, cast_type):
    self.conf[key] = cast_type(self.conf[key])
  
  def forward(self, input):
    contexts, positions, context_len = input[0], input[1], input[2]
    mentions, mention_chars = input[3], input[4]
    type_indexes = input[5]

    mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
    context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

    input_vec = torch.cat((mention_vec, context_vec), dim=1)


class CharEncoder(nn.Module):
    
    def __init__(self, conf):
    
      self.CHARS = ['!', '"', '#', '$', '%', '&', "'", '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', \
      '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',\
      'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd',\
      'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',\
      '{', '}', '~', '·', 'Ì', 'Û', 'à', 'ò', 'ö', '˙', 'ِ', '’', '→', '■', '□', '●', '【', '】', 'の', '・', '一', '（',\
      '）', '＊', '：', '￥', ' ']


      super(CharEncoder, self).__init__()
      conv_dim_input = 100
      filters = 5
      self.char_W = nn.Embedding(len(self.CHARS), conv_dim_input, padding_idx=0)
      self.conv1d = nn.Conv1d(conv_dim_input, conf['char_emb_size'], filters)  # input, output, filter_number

    def forward(self, span_chars):
        char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
        conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
        conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
        cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
        cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
        return cnn_output

class MentionEncoder(nn.Module):

    def __init__(self, conf):
        super(MentionEncoder, self).__init__()
        self.char_encoder = CharEncoder(conf)
        self.attentive_weighted_average = SelfAttentiveSum(conf['emb_size'], 1)
        self.dropout = nn.Dropout(conf['mention_dropout_size'])

    def forward(self, mentions, mention_chars, word_lut):
        mention_embeds = word_lut(mentions)             # batch x mention_length x emb_size

        weighted_avg_mentions, _ = self.attentive_weighted_average(mention_embeds)
        char_embed = self.char_encoder(mention_chars)
        output = torch.cat((weighted_avg_mentions, char_embed), 1)
        return self.dropout(output)

class ContextEncoder(nn.Module):
    def __init__(self, conf):
        self.emb_size = conf['emb_size']
        self.pos_emb_size = conf['positional_emb_size']
        self.rnn_size = conf['context_rnn_size']
        self.hidden_attention_size = 100
        super(ContextEncoder, self).__init__()
        self.pos_linear = nn.Linear(1, self.pos_emb_size)
        self.context_dropout = nn.Dropout(conf['context_dropout'])
        self.rnn = nn.LSTM(self.emb_size + self.pos_emb_size, self.rnn_size, bidirectional=True, batch_first=True)
        self.attention = SelfAttentiveSum(self.rnn_size * 2, self.hidden_attention_size) # x2 because of bidirectional

    def forward(self, contexts, positions, context_len, word_lut, hidden=None):
        """
        :param contexts: batch x max_seq_len
        :param positions: batch x max_seq_len
        :param context_len: batch x 1
        """
        positional_embeds = self.get_positional_embeddings(positions)   # batch x max_seq_len x pos_emb_size
        ctx_word_embeds = word_lut(contexts)                            # batch x max_seq_len x emb_size
        ctx_embeds = torch.cat((ctx_word_embeds, positional_embeds), 2)

        ctx_embeds = self.context_dropout(ctx_embeds)

        rnn_output = self.sorted_rnn(ctx_embeds, context_len)

        return self.attention(rnn_output)

    def get_positional_embeddings(self, positions):
        """ :param positions: batch x max_seq_len"""
        pos_embeds = self.pos_linear(positions.view(-1, 1))                     # batch * max_seq_len x pos_emb_size
        return pos_embeds.view(positions.size(0), positions.size(1), -1)        # batch x max_seq_len x pos_emb_size

    def sorted_rnn(self, ctx_embeds, context_len):
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(ctx_embeds, context_len)
        packed_sequence_input = pack(sorted_inputs, sorted_sequence_lengths, batch_first=True)
        packed_sequence_output, _ = self.rnn(packed_sequence_input, None)
        unpacked_sequence_tensor, _ = unpack(packed_sequence_output, batch_first=True)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

class SelfAttentiveSum(nn.Module):
    """
    Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
    """
    def __init__(self, embed_dim, hidden_dim):
        """
        :param embed_dim: in forward(input_embed), the size will be batch x seq_len x emb_dim
        :param hidden_dim:
        """
        super(SelfAttentiveSum, self).__init__()
        self.key_maker = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key_rel = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.key_output = nn.Linear(hidden_dim, 1, bias=False)
        self.key_softmax = nn.Softmax(dim=1)

    def forward(self, input_embed):     # batch x seq_len x emb_dim
        input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])  # batch * seq_len x emb_dim
        k_d = self.key_maker(input_embed_squeezed)      # batch * seq_len x hidden_dim
        k_d = self.key_rel(k_d)
        if self.hidden_dim == 1:
            k = k_d.view(input_embed.size()[0], -1)     # batch x seq_len
        else:
            k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)
        weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)  # batch x seq_len x 1
        weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, embed_dim
        return weighted_values, weighted_keys
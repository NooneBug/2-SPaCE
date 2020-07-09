from torch.nn import Module
from torch import nn

class Classifier(Module):

  def define_factory(self):
    self.losses_factory = {
        'BCELoss': nn.BCELoss
      }
  

  def __init__(self, config, classes_number):
    self.nametag = 'CLASSIFIER'

    self.cast_params(config[self.nametag])

    self.define_factory()

    super().__init__()

    self.layers = nn.ModuleList()

    prec = self.input_dim

    for dim in self.layer_sizes:
      self.layers.append(nn.Linear(prec, dim).cuda())            
      prec = dim

    self.classification_layer = nn.Linear(prec, classes_number, bias=True).cuda()

    self.sigmoid = nn.Sigmoid()
    self.leaky_relu = nn.LeakyReLU(0.1).cuda()
    self.classification_loss = self.get_classifier_loss(config)()

  def compute_loss(self, pred, true):
    return self.classification_loss(pred, true)

  def get_classifier_loss(self, config):
    loss_class = config[self.nametag]['LOSS']

    # loss_config = config[loss_keyword]['Class']

    loss_manager = self.losses_factory[loss_class]

    return loss_manager


  def forward(self, x):
    
    for i in range(len(self.layers)):
        x = self.leaky_relu(self.layers[i](x))
    classifier_output = self.sigmoid(self.classification_layer(x))
    return classifier_output

  def cast_params(self, config):
    self.layer_sizes = [int(l) for l in config['layers'].split(' ')]
    self.input_dim = int(config['INPUT_SIZE'])
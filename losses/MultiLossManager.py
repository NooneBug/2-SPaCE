from losses.losses import *

class MultiLossManager():
  
  def __init__(self, config):
    self.nametag = 'MultiLossManager'
    self.conf = dict(config[self.nametag])

    # self.names = self.conf['Names']

    self.define_factory()

    loss_classes = self.get_classes(config)

    self.losses = {k:l(config) for k, l in loss_classes.items()}

  def compute_loss(self, true_vectors, projection_output):
    losses_values = {k:l.compute_loss(true_vectors[k], projection_output[k]) for k, l in self.losses.items()}
    print('losses_values: {}'.format(losses_values))

  def get_classes(self, conf):
    self.classes_names = conf[self.nametag]['LOSSES'].split(' ')
    classes = {}
    for name in self.classes_names:
      classes[conf[name]['NAME']] = self.loss_factory[conf[name]['CLASS']]
    return classes

  def define_factory(self):
    self.loss_factory = {
      'cosine_dissimilarity': CosineDissimilarityLoss,
      'hyperbolic_distance': HyperbolicDistanceLoss,
      'normalized_hyperbolic_distance': 'NHYPD',
      'regularized_hyperbolic_distance': 'RHYPD',
      'hyperboloid_distance' : 'LORENTZD',
      'multilabel_Minimum_Normalized_Poincare': 'NHMML',
      'multilabel_Minimum_Poincare': MultilabelMinimumPoincare,
      'multilabel_Minimum_Cosine': MultilabelMinimumCosine,
      'multilabel_Average_Poincare': MultilabelAveragePoincare,
      'multilabel_Average_Cosine': MultilabelAverageCosine,
    }

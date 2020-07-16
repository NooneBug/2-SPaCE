from abc import ABC

import torch

class Loss(ABC):

  def __init__(self, config):
    pass

  def compute_loss(self, true, pred):
    pass

class CosineDissimilarityLoss(Loss):

  def __init__(self, config):
    pass

  def compute_loss(self, true, pred):
    cossim = torch.nn.CosineSimilarity(dim = 1)
    return 1 - cossim(true, pred)

class HyperbolicDistanceLoss(Loss):
  
  def __init__(self, config):
    pass

  def acosh(self, x):
        return torch.log(x + torch.sqrt(x**2 - 1))

  def compute_loss(self, true, pred):
    numerator = 2 * torch.norm(true - pred, dim = 1)**2

    pred_norm = torch.norm(pred, dim = 1)**2
    true_norm = torch.norm(true, dim = 1)**2

    left_denom = 1 - pred_norm
    right_denom = 1 - true_norm

    denom = left_denom * right_denom

    frac = numerator/denom

    acos = self.acosh(1  + frac)
    
    return acos

class MultilabelMinimumPoincare(HyperbolicDistanceLoss):
  def compute_loss(self, true, pred):
    for i in range(true.shape[1]):
      t = torch.index_select(true, 1, index=torch.tensor([i]).cuda()).squeeze()

      if len(t.shape) == 1:
        t = t.view(1, t.shape[0])

      loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
      
      if i == 0:
        min_tensor = loss
      else:
        catted = torch.cat((loss, min_tensor), dim = 0)
        min_tensor, _ = torch.min(catted, dim=0)
        min_tensor = min_tensor.view(1, true.shape[0])

    return min_tensor

class MultilabelAveragePoincare(HyperbolicDistanceLoss):
  def compute_loss(self, true, pred):
    for i in range(true.shape[1]):
      t = torch.index_select(true, 1, index=torch.tensor([i]).cuda()).squeeze()

      if len(t.shape) == 1:
        t = t.view(1, t.shape[0])
      
      loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
      if i == 0:
        mean_tensor = loss
      else:
        mean_tensor = torch.cat((loss, mean_tensor), dim = 0)

    return torch.mean(mean_tensor, dim = 0)

class MultilabelMinimumCosine(CosineDissimilarityLoss):
  def compute_loss(self, true, pred):
    for i in range(true.shape[1]):
      t = torch.index_select(true, 1, index=torch.tensor([i]).cuda()).squeeze()

      if len(t.shape) == 1:
          t = t.view(1, t.shape[0])
      
      loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
      
      if i == 0:
        min_tensor = loss
      else:
        catted = torch.cat((loss, min_tensor), dim = 0)
        min_tensor, _ = torch.min(catted, dim=0)
        min_tensor = min_tensor.view(1, true.shape[0])

    return min_tensor

class MultilabelAverageCosine(CosineDissimilarityLoss):
  def compute_loss(self, true, pred):
    for i in range(true.shape[1]):
      t = torch.index_select(true, 1, index=torch.tensor([i]).cuda()).squeeze()

      if len(t.shape) == 1:
        t = t.view(1, t.shape[0])
      
      loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
      
      if i == 0:
        mean_tensor = loss
      else:
        mean_tensor = torch.cat((loss, mean_tensor), dim = 0)

    return torch.mean(mean_tensor, dim = 0)
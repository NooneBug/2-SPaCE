# %%
from losses.losses import HyperbolicDistanceLoss
import torch
h = HyperbolicDistanceLoss('config')

a = torch.tensor([[0.2, 0.2]])

h.compute_loss(a, a)
# %%

import torch
from torch import optim

params = 1
optimizer = torch.optim.Adadelta(params,lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None,maximize=False, differentiable=False)

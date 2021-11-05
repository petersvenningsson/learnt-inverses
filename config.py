import torch

n_components = 10
sample_gitter = torch.Tensor([i for i in range(300)]).unsqueeze(1).unsqueeze(1)

param_interval = {'k':{'min': 0, 'max': 100}}
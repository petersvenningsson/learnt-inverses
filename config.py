import torch

max_components = 10

# May be more realistic to generate k values as uniform distributin in log scale
param_interval = {'k':{'min': 0.01, 'max': 1}}
sample_gitter = torch.linspace(0.1, 1/param_interval['k']['min'], 300 ).unsqueeze(1).unsqueeze(1)

weighting_scheme = 'dirichlet'
import torch
import numpy as np

max_components = 10

# May be more realistic to generate k values as uniform distributin in log scale
param_interval = {'k':{'min': 0.01, 'max': 1}}

# create sampling gitter (equispaced in logarithmic scale)
sampling_interval = (param_interval['k']['min'], 1/param_interval['k']['min'])

sample_gitter = torch.linspace(np.log10(sampling_interval[0]), np.log10(sampling_interval[1]), 300 ).unsqueeze(1).unsqueeze(1)
sample_gitter = 10**sample_gitter
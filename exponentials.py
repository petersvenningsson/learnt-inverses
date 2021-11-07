import torch
from numpy import random
import numpy as np

from config import max_components, sample_gitter, param_interval, weighting_scheme

def sample_linear_combination(batch_size, device):
    # Maybe more realistic to draw components from dirichlet distribution
    # import numpy as np, numpy.random
    # print np.random.dirichlet(np.ones(10),size=1)
    # First argument decides if we have a dominating term
    # https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1


    components = torch.randint(low=1, high=max_components, size=(batch_size,))

    if weighting_scheme == 'uniform':
        w = torch.rand((batch_size, max_components))

        for i, c in enumerate(components):
            w[i, c:] = 0
            w = (w/w.sum(axis=1).unsqueeze(1)).to(device)

    elif weighting_scheme == 'dirichlet':
        scale_factors = random.rand(batch_size)*1 + 1/100
        _w = np.array([random.dirichlet(np.ones(max_components)*s, size = 1) for s in scale_factors])
        w = torch.Tensor(_w).squeeze().to(device)

    k = (torch.rand((batch_size, max_components)) * (param_interval['k']['max'] - param_interval['k']['min']) + param_interval['k']['min']).to(device)
    
    return linear_combination(k, w, device), k, w


def linear_combination(k, w, device):

    sample = sample_gitter.clone().to(device)

    return (torch.exp(-k*sample) * w).sum(axis=2).T


if __name__ == '__main__':
    time_series, params = sample_linear_combination(batch_size = 58)
    time_series = linear_combination(params[:10,:],params[10:,:])
    print('s')
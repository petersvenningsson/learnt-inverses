import torch

from config import n_components, sample_gitter, param_interval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_linear_combination(batch_size):
    # Maybe more realistic to draw components from dirichlet distribution
    # import numpy as np, numpy.random
    # print np.random.dirichlet(np.ones(10),size=1)
    # First argument decides if we have a dominating term
    # https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
    w = torch.rand((batch_size, n_components))
    w = (w/w.sum(axis=0)).to(device)
    k = (torch.rand((batch_size, n_components)) * (param_interval['k']['max'] - param_interval['k']['min']) + param_interval['k']['min']).to(device)
    
    return linear_combination(k, w), torch.cat((k,w), 0)


def linear_combination(k, w):

    sample = sample_gitter.clone().to(device)

    return (torch.exp(-k*sample) * w).sum(axis=2).T


# def sample_kernel(k):
#     return 


if __name__ == '__main__':
    time_series, params = sample_linear_combination(batch_size = 58)
    time_series = linear_combination(params[:10,:],params[10:,:])
    print('s')
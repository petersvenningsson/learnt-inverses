import itertools
import csv

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from model import NeuralNet
import exponentials
from config import max_components, sample_gitter
import matplotlib.pyplot as plt

# Notes: Qualatative performance decreased with a sampling gitter eqvidistant in logarithmic scale

# Grid search:
# Learning rate, hidden size, n_layers, if output is scaled to interval, just normal weight L2 regularization

# TODO grid search
# Add regularization - add later if we encounter problems with ofysikaliska values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters 
num_params = 20
num_epochs = 50000
batch_size = 128

# Define grid search
hidden_size_grid = [256, 512, 1024, 2048]
hidden_size_grid.reverse()
n_layers_grid = [3, 6, 9, 12, 15]
n_layers_grid.reverse()
learning_rate_grid = [0.001, 0.0001, 0.00001]
normlized_grid = [True, False]
objective_grid = [nn.MSELoss, nn.L1Loss]
weighting_scheme_grid = ['dirichlet', 'uniform']

stds = {}
means = {}
x_vals = {}
for weighting_scheme in weighting_scheme_grid:
    # Draw standardization data
    x, _, _ = exponentials.sample_linear_combination(500000,'cpu', weighting_scheme)
    stds[weighting_scheme] = x.std(axis=0).to(device)
    means[weighting_scheme] = x.mean(axis=0).to(device)

    # Draw validation data
    x_val, _, _ = exponentials.sample_linear_combination(500000, device, weighting_scheme)
    x_vals[weighting_scheme] = x_val

file = open('log.csv', 'w')
logwrite = csv.writer(file)
logwrite.writerow(['hidden_size', 'n_layers', 'learning_rate', 'normalized_output', 'Objective', 'loss', 'moving_average', 'weighting_scheme'])

for (hidden_size, n_layers, learning_rate, normalized_output, Objective, weighting_scheme) in itertools.product(hidden_size_grid, n_layers_grid, learning_rate_grid, normlized_grid, objective_grid, weighting_scheme_grid):

    std = stds[weighting_scheme]
    mean = means[weighting_scheme]
    x_val = x_vals[weighting_scheme]



    model = NeuralNet(sample_gitter.shape[0], hidden_size, num_params, n_layers, normalized_output = normalized_output).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    objective = Objective()

    current_best = np.array([1e20])
    _moving_average = np.array([1e20])
    moving_average = np.array([1e20])

    for epoch in range(num_epochs):
        x, params, weights = exponentials.sample_linear_combination(batch_size, device, weighting_scheme)
        x = x.to(device)

        predicted_params, predicted_weights = model((x-mean)/std)
        predicted_x = exponentials.linear_combination(predicted_params, predicted_weights, device)

        loss = objective(predicted_x, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch%500 == 0:
            with torch.no_grad():
                predicted_params, predicted_weights = model((x_val-mean)/std)

                predicted_x = exponentials.linear_combination(predicted_params, predicted_weights, device)
                
                loss = objective(predicted_x, x_val)
                if _moving_average:
                    _moving_average = _moving_average*0.5 + loss.item()*0.5
                else:
                    _moving_average = loss.item()

                if loss.item() < current_best:
                    current_best = loss.item()
                    moving_average = _moving_average
                    

                print(f"The validation MSE is {loss.item()}, current best = {current_best}, moving average: {moving_average}")
    
    data = [hidden_size, n_layers, learning_rate, normalized_output, Objective.__name__, weighting_scheme]
    row = [str(d) for d in data]
    row.extend([current_best, moving_average])
    logwrite.writerow(row)
file.close()
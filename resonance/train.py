import torch
import torch.nn as nn

from model import NeuralNet
from config import max_components, sample_gitter
import matplotlib.pyplot as plt

import sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO, params --> signal, write wrapper for batch in sample.py

# Define Hyper-parameters 
hidden_size = 2048
num_params = 10*max_components # 9 parameters and the weights w
num_epochs = 1000000
batch_size = 128
learning_rate = 0.0001
n_layers = 15
l2_reg = 0

model = NeuralNet(sample_gitter.shape[0], hidden_size, num_params, n_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)  
mse = nn.MSELoss()
l1 = nn.L1Loss()

# # Draw standardization data
x, _, _ = sample.draw(500000,'cpu', 'dirichlet')
std = x.std(axis=0).to(device)
mean = x.mean(axis=0).to(device)

# Draw validation data
x_val, params_val, weights_val = sample.draw(500000, device, 'dirichlet')

current_best = 1e20

for epoch in range(num_epochs):
    x, params, weights = sample.draw(batch_size, device,'dirichlet')
    x = x.to(device)

    predicted_params, predicted_weights = model((x-mean)/std)
    predicted_x = exponentials.linear_combination(predicted_params, predicted_weights, device)

    loss = l1(predicted_x, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch%500 == 0:
        with torch.no_grad():
            predicted_params, predicted_weights = model((x_val-mean)/std)

            predicted_x = exponentials.linear_combination(predicted_params, predicted_weights, device)
            
            loss = l1(predicted_x, x_val)
            # Two choices for limiting predicted params to realistic intervals
            ## 1. Add loss regularization loss to keep parameters inside interval
            ## A little bit of loss based on the generated parameters
            if loss < current_best:
                current_best = loss
            plt.plot(sample_gitter.cpu().numpy().squeeze(), (predicted_x)[0,:].cpu().numpy())
            plt.plot(sample_gitter.cpu().numpy().squeeze(), x_val[0,:].cpu().numpy())
            plt.legend(['predicted signal', 'true signal'])
            plt.xlabel('sampling gitter')
            plt.ylabel('y')
            plt.title('Plot of y(sampling gitter)')
            plt.savefig('validation.png')
            plt.clf()

            if 0:
                print("Predictions")
                for i in range(max_components):
                    print(f"Weight: {predicted_weights[0,i]:.5f}, Parameter {predicted_params[0,i]:.5f}")
                
                print('True values')
                for i in range(max_components):
                    print(f"Weight: {weights_val[0,i]:.5f}, Parameter {params_val[0,i]:.5f}")          

            print(f"The validation MSE is {loss}, current best = {current_best}")
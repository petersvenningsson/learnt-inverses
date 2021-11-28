import torch
import torch.nn as nn

from model import NeuralNet
from config import max_components
import matplotlib.pyplot as plt

import sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 809

# TODO
# Model interface for multiple compnents is completed. Time to add multiple components to sample.py. Remember, mistakes were done with reshape. Be careful check check check

# Define Hyper-parameters 
hidden_size = 2048
num_params = (7 + 2*2 + 1)*max_components # 7 scalar paramters and 2 periodic parameters and 1 weight
num_epochs = 1000000    
batch_size = 32
learning_rate = 0.000001
n_layers = 15
l2_reg = 0
val_size = 50000

model = NeuralNet(input_size, hidden_size, num_params, n_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)  
# l1 = nn.MSELoss()
# l1 = nn.L1Loss()
l1 = nn.HuberLoss(delta=10)

# # Draw standardization data
x, _, = sample.draw('cpu', 100000, 'dirichlet')
std = x.std(axis=0).to(device)
mean = x.mean(axis=0).to(device)


# Draw validation data
x_val, dist_val = sample.draw(device, val_size, 'dirichlet')

current_best = 1e20

#with torch.autograd.detect_anomaly():
for epoch in range(num_epochs):

    x, dist_train = sample.draw(device, batch_size, 'dirichlet')
    x = x.to(device)

    predicted_dist = model((x-mean)/std)
    predicted_x = sample.generate_signal(predicted_dist, batch_size, device)

    loss = l1(predicted_x, x)

    # print(f'Training loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch%500 == 0:
        with torch.no_grad():
            predicted_dist = model((x_val-mean)/std)

            predicted_x = sample.generate_signal(predicted_dist, batch_size, device)

            loss = l1(predicted_x, x_val)
            plt.plot(predicted_x[0,:].cpu().detach())
            plt.plot(x_val[0,:].cpu().detach())
            plt.legend(['predicted', 'real'])
            plt.savefig('image2.png')
            plt.clf()

            if loss < current_best:
                current_best = loss
            # plt.plot(sample_gitter.cpu().numpy().squeeze(), (predicted_x)[0,:].cpu().numpy())
            # plt.plot(sample_gitter.cpu().numpy().squeeze(), x_val[0,:].cpu().numpy())
            # plt.legend(['predicted signal', 'true signal'])
            # plt.xlabel('sampling gitter')
            # plt.ylabel('y')
            # plt.title('Plot of y(sampling gitter)')
            # plt.savefig('validation.png')
            # plt.clf()

            # if 0:
            #     print("Predictions")
            #     for i in range(max_components):
            #         print(f"Weight: {predicted_weights[0,i]:.5f}, Parameter {predicted_params[0,i]:.5f}")
                
            #     print('True values')
            #     for i in range(max_components):
            #         print(f"Weight: {weights_val[0,i]:.5f}, Parameter {params_val[0,i]:.5f}")          

            print(f"The validation MSE is {loss}, current best = {current_best}")
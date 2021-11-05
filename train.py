import torch
import torch.nn as nn

from model import NeuralNet
import exponentials
from config import n_components, sample_gitter

# TODO Parametrize network depth ect, normalize input, grid search
# Loss function based on the signal instead of the parameters
# Predict value small value which is mapped to the value interval
# Add regularization

# TODO Remove gradient on validtion
# TODO Use softmax on weights
# TODO Take second power of params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters 
input_size = 784
hidden_size = 1024
num_params = 20
num_epochs = 1000000
batch_size = 100
learning_rate = 0.001

model = NeuralNet(sample_gitter.shape[0], hidden_size, num_params).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
mse = nn.MSELoss()
l1 = nn.L1Loss()

x, labels_val = exponentials.sample_linear_combination(batch_size*100)
x_val = x.to(device)
labels_val = labels_val.to(device)

current_best = 1000000

for epoch in range(num_epochs):
    x, params = exponentials.sample_linear_combination(batch_size)
    x = x.to(device)
    params = params.to(device)
    y = model(x)
    predicted_x = exponentials.linear_combination(y[:,:10], y[:,10:])

    loss = l1(predicted_x, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100 == 0:
        y = model(x_val)

        predicted_x = exponentials.linear_combination(y[:,:10], y[:,10:])
        loss = l1(predicted_x, x_val)
        if loss < current_best:
            current_best = loss
        
        print(f"The validation MSE is {loss}, current best = {current_best}")
import torch
import torch.nn as nn

from config import max_components, parameters, parameter_interval

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, normalized_output = False):
        super(NeuralNet, self).__init__()
        assert n_layers > 2, "Please specify more than 2 layers"
        self.n_layers = n_layers - 2
        self.normalized_output = normalized_output

        self.input_layer = nn.Linear(input_size, hidden_size)

        layers = []
        for _ in range(self.n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_layers = nn.ModuleList(layers)

        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        out = self.input_layer(x)

        for hidden_layer in self.hidden_layers:
            out = hidden_layer(out)
            out = self.relu(out)

        out = self.relu(out)
        out = self.output_layer(out)

        _params, weights = out[:,:-max_components], out[:,-max_components:]

        # Normalize weights
        weights = weights**2
        # weights = self.softmax(weights)
        weights = weights/weights.sum(1).unsqueeze(1)

        # # Map to parameter interval
        # if self.normalized_output:
        #     params = params*(param_interval['k']['max']-param_interval['k']['min'])/2 + (param_interval['k']['max']-param_interval['k']['min'])/2

        dist = {}
        for j, parameter in enumerate(parameters):
            dist[parameter] = _params[:,j*10:(j+1)*10].unsqueeze(1)

        dist['w'] = weights.unsqueeze(1)

        # scale to interval
        for parameter in parameter_interval.keys():
            dist[parameter] = parameter_interval[parameter][0]*(parameter_interval[parameter][1]/parameter_interval[parameter][0])**dist[parameter]

        return dist
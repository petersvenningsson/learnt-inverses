import torch
import torch.nn as nn

from config import max_components, bounded_parameters, cyclic_parameters, parameter_interval

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, normalized_output = False):
        super(NeuralNet, self).__init__()
        assert n_layers > 2, "Please specify more than 2 layers"
        self.n_layers = n_layers - 2
        self.output_size = output_size
        self.n_parameters = int(output_size/max_components)

        self.input_layer = nn.Linear(input_size, hidden_size)

        layers = []
        for _ in range(self.n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_layers = nn.ModuleList(layers)

        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.input_layer(x)

        for hidden_layer in self.hidden_layers:
            out = hidden_layer(out)
            out = self.relu(out)

        out = self.sigmoid(self.output_layer(out))

        params = out.reshape(-1, max_components, self.n_parameters)

        dist = {}
        
        index = 0
        for j, parameter in enumerate(bounded_parameters):
            dist[parameter] = params[:,:,index].unsqueeze(2)
            index += 1

        # scale bounded variables to interval
        for parameter in parameter_interval.keys():
            dist[parameter] = parameter_interval[parameter][0]*(parameter_interval[parameter][1]/parameter_interval[parameter][0])**dist[parameter]

        for parameter in cyclic_parameters:

            cos = params[:,:,index] - 0.5
            index = index + 1

            sin = params[:,:,index] - 0.5
            index = index + 1

            if parameter == 'phi':
                dist[parameter] = ((torch.atan(sin/cos) + torch.pi/2)*2)
            
            if parameter == 'theta':
                dist[parameter] = (torch.atan(sin/cos) + torch.pi/2)

        dist['w'] = (params[:,:,index]/(params[:,:,index].sum(1).unsqueeze(1))).unsqueeze(2)

        return dist
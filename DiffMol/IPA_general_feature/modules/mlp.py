
import torch.nn as nn
import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, bn=False):
        '''
        3 layer MLP

        Args:
            input_dim: # input layer nodes
            hidden_dim: # hidden layer nodes
            output_dim: # output layer nodes
            activation: activation function
            layer_norm: bool; if True, apply LayerNorm to output
        '''

        # init superclass and hidden/ output layers
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

        self.bn = bn
        if self.bn:
            self.bn = nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.997)

        # init activation function reset parameters
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):

        # reset model parameters
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x):

        # forward prop x
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        x = self.lin3(x)

        return x

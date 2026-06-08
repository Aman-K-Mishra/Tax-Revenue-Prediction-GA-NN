from typing import List
import torch.nn as nn


class FeedForwardNN(nn.Module):
    """
    Simple fully connected network for regression.
    Architecture is defined by GA chromosome: list of layer sizes + activations.
    """

    def __init__(self, input_dim: int, layers: List[int], activations: List[str]):
        super().__init__()

        modules = []
        prev_dim = input_dim

        for size, act in zip(layers, activations):
            modules.append(nn.Linear(prev_dim, size))

            if act == "relu":
                modules.append(nn.ReLU())
            elif act == "tanh":
                modules.append(nn.Tanh())
            elif act == "sigmoid":
                modules.append(nn.Sigmoid())

            prev_dim = size

        # final regression output layer
        modules.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

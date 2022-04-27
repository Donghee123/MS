import torch
import torch.nn as nn

class Network(torch.nn.Module):
    def __init__(self, inputSize = int, outputSize = int):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        in_size = inputSize
        layers = [
            nn.Linear(in_size, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, outputSize),
        ]
        self.laysers = nn.Sequential(*layers)

    def forward(self, A0):
        x = self.laysers(A0)
        return x
import torch
import torch.nn as nn
from .atomic_layer import AtomicLayer


class AtomicModel(nn.Module):
    def __init__(self, size, params):
        super().__init__()
        self.size = size
        self.params = params
        self.atomic_model = nn.Sequential(*[
            AtomicLayer(self.size, **param) for param in self.params
        ])

    def forward(self, input):
        return self.atomic_model(input).mean(dim=[1,2,3,4])
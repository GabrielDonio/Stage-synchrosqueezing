import torch
from abc import ABC, abstractmethod

class Transformation(torch.nn.Module, ABC):
    def __init__(self, M, eps=1e-6):
        super().__init__()
        self.M = M
        self.eps = eps

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
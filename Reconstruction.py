import torch
from abc import ABC, abstractmethod

class Reconstruction(torch.nn.Module, ABC):
    """
    Base class for reconstruction methods.

    Args:
        M (int): Number of frequency bins.
        eps (float): Threshold for the reconstruction.
    """
    def __init__(self, M, eps=1e-6):
        super().__init__()
        self.M = M
        self.eps = eps
    @abstractmethod
    def forward(self, tfr):
        raise NotImplementedError
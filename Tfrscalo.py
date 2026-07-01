import torch
import numpy as np
from Transformation import Transformation
from mmaxis import a_axis

class Tfrscalo(Transformation):
    """
    Implementation of the time-frequency representation using a continuous wavelet transform with a Morlet wavelet.
    Args:
        M (int): Number of frequency bins.
        T (float): Total time duration.
        Ts (float): Sampling period.
        w0 (float): Central frequency of the Morlet wavelet.
        eps (float): Threshold for the synchrosqueezing transform.
        gamma_K (float): Threshold for the Gaussian window.
        as_range (list): Range of scales for the wavelet transform.
        is_freq (int): Whether to use frequency axis or time axis for the scales.
    Returns:
        tfr (torch.Tensor): Time-frequency representation.
    """
    def __init__(self, M, T, Ts, w0, eps=1e-6, gamma_K=1e-5, as_range=[0.05, 1.5], is_freq=0):
        super().__init__(M, eps)

        self.T = T          
        self.Ts = Ts        
        self.w0 = w0        
        self.gamma_K = gamma_K
        self.as_range = as_range
        self.is_freq = is_freq
        self.scales = a_axis(M, as_range, method=is_freq)

        self.sqrt_pi = torch.sqrt(torch.tensor(torch.pi))
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        sqrt_log_gamma = torch.sqrt(2 * torch.log(torch.tensor(1.0 / self.gamma_K)))

        for n in range(N):
            for m in range(self.M):
                a = self.scales[m]
                
                K = int(torch.round(sqrt_log_gamma * (self.T / self.Ts) * a).item())
                
                k_min = min(n, K)
                k_max = min(N - 1 - n, K)
                
                k = torch.arange(-k_min, k_max + 1, device=device)
                tau = k * self.Ts
                
                x_slice = x[n + k]
                
                gauss = torch.exp(- (tau ** 2) / (2 * (self.T * a) ** 2))
                
                osc = torch.exp(-1j * self.w0 * tau / a)
                
                norm = self.Ts / torch.sqrt(torch.abs(a) * self.T * self.sqrt_pi)
                
                tfr[m, n] = norm * torch.sum(x_slice * gauss * osc)
                
        return tfr


import torch
from Transformation import Transformation
class Tfrgab(Transformation):
    """
    implementation of the time-frequency representation with a gabor window.

    Args:
        M (int): Number of frequency bins.
        eps (float): Threshold for the time-frequency representation.
        L (int): Parameter for the Gabor window.
        gamma_K (float): threshold for the Parameter for the Gabor window.
    """
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.L = L
        self.gamma_K = gamma_K

        self.K = 2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))
        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.B = -1j * 2 * torch.pi / self.M
        self.C = -1 / (2 * self.L**2)
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device      
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        mm = torch.arange(0, self.M, device=device)
        
        for n in range(N):
            k_min = min(n, int(torch.round(self.K / 2).item()))
            k_max = min(N - 1 - n, int(torch.round(self.K / 2).item()))

            k = torch.arange(-k_min, k_max + 1, device=device)
            k2 = k ** 2

            g = self.A * torch.exp(self.C * k2) # fenetre de gabor
            for m in range(self.M):
                exp_B_mm_k = torch.exp(self.B * mm[m] * k)
                exp_B_mm_nn = torch.exp(self.B * mm[m] * n)
                
                x_slice = x[n + k]
                
                tfr[m, n] = exp_B_mm_nn * torch.sum(x_slice * g * exp_B_mm_k)
                
        return tfr

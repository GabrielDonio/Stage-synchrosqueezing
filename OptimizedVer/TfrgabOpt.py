import torch
from Transformation import Transformation
from mmaxis import m_axis

class Tfrgab(Transformation):
    """
    Implementation of the time-frequency representation with a Gabor window (Optimized).
    """
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K

        self.K_val = int(torch.round(2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))).item())
        self.half_K = self.K_val // 2

        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.B = -1j * 2 * torch.pi / self.M
        self.C = -1 / (2 * self.L**2)
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device      
        
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        mm = m_axis(self.M, device=device).view(-1, 1)
        
        for n in range(N):
            k_min = min(n, self.half_K)
            k_max = min(N - 1 - n, self.half_K)

            k = torch.arange(-k_min, k_max + 1, device=device)
            k2 = k ** 2

            g = self.A * torch.exp(self.C * k2)
            x_slice = x[n + k]

            exp_B_mm_k = torch.exp(self.B * mm * k)

            exp_B_mm_n = torch.exp(self.B * mm * n)

            tfr[:, n] = exp_B_mm_n.squeeze() * torch.sum(x_slice * g * exp_B_mm_k, dim=1)
                
        return tfr
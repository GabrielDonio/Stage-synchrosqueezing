import torch
from Transformation import Transformation
from mmaxis import m_axis

class TfrWin(Transformation):
    """
    Implementation of the Short-Time Fourier Transform (STFT) with a custom window.
    """
    def __init__(self, M, window, eps=1e-6):
        super().__init__(M, eps)
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]
        
        self.omega_coef = -1j * 2 * torch.pi / self.M

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device

        w_v = self.window.to(device)
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        mm = m_axis(self.M, device=device).view(-1, 1) 

        center = self.len_win // 2
        left = center
        right = self.len_win - center - 1

        for n in range(N):
            k_min = min(n, left)
            k_max = min(N - 1 - n, right)

            k = torch.arange(-k_min, k_max + 1, device=device)
            
            w_slice = w_v[center - k_min : center + k_max + 1]
            x_slice = x[n + k]

            exp_B_mm_k = torch.exp(self.omega_coef * mm * k)

            exp_phase_n = torch.exp(self.omega_coef * mm * n)[:, 0]

            tfr[:, n] = exp_phase_n * torch.sum(x_slice * w_slice * exp_B_mm_k, dim=1)

        return tfr
import torch
import matplotlib.pyplot as plt
from Transformation.Transformation import Transformation
from mmaxis import m_axis

class TfrWin(Transformation):
    def __init__(self, M, window, eps=1e-6):
        super().__init__(M, eps)
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        mm = m_axis(self.M, device=device).view(-1, 1) 

        center = self.len_win // 2
        left = center
        right = self.len_win - center - 1

        for n in range(N):
            k_min = min(n, left)
            k_max = min(N - 1 - n, right)

            k = torch.arange(-k_min, k_max + 1, device=device)
            w = self.window[center - k_min : center + k_max + 1].to(device)
            x_slice = x[n + k]

            exp_B_mm_k = torch.exp(-1j * 2 * torch.pi * mm * k / self.M)

            exp_phase_n = torch.exp(-1j * 2 * torch.pi * mm * n / self.M)

            tfr[:, n] = exp_phase_n.squeeze() * torch.sum(x_slice * w * exp_B_mm_k, dim=1)

        return tfr
import torch
import matplotlib.pyplot as plt


class TfrWin(torch.nn.Module):
    """
    implementation of the time-frequency representation with a given window.

    Args:
        M (int): Number of frequency bins.
        window (torch.Tensor): The window function.
        eps (float): Threshold for the time-frequency representation.
    Returns:
        tfr (torch.Tensor): Time-frequency representation.
    """
    def __init__(self, M, window, eps=1e-6):
        super().__init__()
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32, device=window.device).reshape(-1)  # window
        self.len_win = window.shape[0]
        self.eps = eps

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        mm = torch.arange(0, self.M, device=device)

        center = self.len_win // 2
        left = center
        right = self.len_win - center - 1

        for n in range(N):
            k_min = min(n, left)
            k_max = min(N - 1 - n, right)

            k = torch.arange(-k_min, k_max + 1, device=device)
            w = self.window[center - k_min : center + k_max + 1]
            x_slice = x[n + k]

            for m in range(self.M):
                exp_B_mm_k = torch.exp(-1j * 2 * torch.pi * mm[m] * k / self.M)
                tfr[m, n] = torch.sum(x_slice * w * exp_B_mm_k)

        return tfr


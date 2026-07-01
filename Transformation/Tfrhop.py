import torch
from Transformation.Transformation import Transformation
from mmaxis import m_axis

class Tfrhop(Transformation):
    def __init__(self, M, window, hop_length, eps=1e-6):
        super().__init__(M, eps)
        self.hop_length = hop_length
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]
        self.omega_coef = -1j * 2 * torch.pi / self.M

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device

        w_v = self.window.to(device)
        time_indices = list(range(0, N, self.hop_length))
        num_frames = len(time_indices)

        tfr = torch.zeros((self.M, num_frames), dtype=torch.complex64, device=device)

        mm = m_axis(self.M, device=device).view(-1, 1)

        center = self.len_win // 2
        left = center
        right = self.len_win - center - 1

        for frame_idx, n in enumerate(time_indices):
            k_min = min(n, left)
            k_max = min(N - 1 - n, right)

            k = torch.arange(-k_min, k_max + 1, device=device)
            w_slice = w_v[center - k_min : center + k_max + 1]
            x_slice = x[n + k]

            exp_B_mm_k = torch.exp(self.omega_coef * mm * k)
            exp_phase_n = torch.exp(self.omega_coef * mm * n)[:, 0]

            tfr[:, frame_idx] = exp_phase_n * torch.sum(x_slice * w_slice * exp_B_mm_k, dim=1)

        return tfr
import torch
from Reconstruction.Reconstruction import Reconstruction

class Stft_rechop(Reconstruction):
    def __init__(self, M, window, hop_length, eps=1e-6):
        super().__init__(M, eps)
        self.M = M
        self.hop_length = hop_length
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]

    def forward(self, tfr, original_N=None):
        tfr = torch.as_tensor(tfr)
        device = tfr.device
        M, num_frames = tfr.shape

        center = self.len_win // 2
        h0 = self.window[center].to(device)

        if torch.abs(h0) < self.eps:
            raise ValueError("h(0) est nul")

        if original_N is None:
            original_N = (num_frames - 1) * self.hop_length + 1

        s_hat = torch.zeros(original_N, dtype=torch.complex64, device=device)

        for frame_idx in range(num_frames):
            n = frame_idx * self.hop_length
            if n >= original_N:
                break
            
            s_hat[n] = torch.sum(tfr[:, frame_idx]) / (self.M * h0)

        return s_hat
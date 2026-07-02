import torch
from Transformation.Transformation import Transformation

class TfrWin(Transformation):
    """
    Implementation of the Short-Time Fourier Transform (STFT) with a custom window.
    """
    def __init__(self, M, window, eps=1e-6):
        super().__init__(M, eps)
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]

        self.center = self.len_win // 2
        self.left = self.center
        self.right = self.len_win - self.center - 1

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device

        w_v = self.window.to(device)
        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.left, self.right), mode='reflect')
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        x_frames = x_padded.unfold(0, self.len_win, 1)[:N]

        tfr_segments = x_frames * w_v

        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t() 

        m_vec = torch.arange(self.M, device=device).view(-1, 1)
        n_vec = torch.arange(N, device=device).view(1, -1)
        
        phase_correction = torch.exp(1j * 2 * torch.pi / self.M * m_vec * (self.left - n_vec))
        
        tfr = tfr * phase_correction

        return tfr
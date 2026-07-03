import torch
from Transform.Transformation import Transformation


class TfrWin(Transformation):
    def __init__(self, M, window, hop_length=1, eps=1e-6):
        super().__init__(M, eps)
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]
        self.hop_length = hop_length

        self.center = self.len_win // 2
        self.left = self.center
        self.right = self.len_win - self.center - 1

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device

        w_v = self.window.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.left, self.right), mode="reflect")
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, N, self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]
        tfr_segments = x_frames * w_v

        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t()

        m_vec = torch.arange(self.M, device=device).view(-1, 1)
        n_vec_row = n_vec.view(1, -1)

        phase_correction = torch.exp(
            1j * 2 * torch.pi / self.M * m_vec * (self.left - n_vec_row)
        )

        return tfr * phase_correction
    def rec(self, tfr):

        device = tfr.device
        M, num_frames = tfr.shape
        w_v = self.window.to(device)

        N_target = getattr(self, 'N_input', (num_frames - 1) * self.hop_length + 1)

        n_vec = torch.arange(0, num_frames * self.hop_length, self.hop_length, device=device)[:num_frames]
        m_vec = torch.arange(M, device=device).view(-1, 1)
        phase_correction = torch.exp(
            1j * 2 * torch.pi / M * m_vec * (self.left - n_vec.view(1, -1))
        )
        tfr_origin = tfr / phase_correction

        tfr_segments = torch.fft.ifft(tfr_origin.t(), n=M, dim=1).real
        tfr_segments = tfr_segments[:, :self.len_win]

        if self.hop_length == 1:
            x_at_t = tfr_segments[:, self.left]

            h_0 = w_v[self.left]

            if torch.abs(h_0) < self.eps:
                raise ValueError("Le centre de la fenêtre h(0) est nul ou trop proche de 0, reconstruction impossible via cette formule.")
                
            x_reconstructed = x_at_t / h_0
            
            return x_reconstructed[:N_target]

        else:
            len_padded_output = (num_frames - 1) * self.hop_length + self.len_win
            x_reconstructed = torch.zeros(len_padded_output, device=device, dtype=torch.float32)
            window_sum = torch.zeros(len_padded_output, device=device, dtype=torch.float32)

            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.len_win
                x_reconstructed[start:end] += tfr_segments[i] * w_v
                window_sum[start:end] += w_v ** 2

            mask = window_sum > self.eps
            x_reconstructed[mask] /= window_sum[mask]

            return x_reconstructed[self.left : self.left + N_target]
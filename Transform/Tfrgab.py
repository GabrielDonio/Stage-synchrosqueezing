import torch
from Transform.Transformation import Transformation


class Tfrgab(Transformation):
    """
    Gabor time-frequency representation.
    Default hop_length=1 reproduces the base behavior.
    """
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4, hop_length=1):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K
        self.hop_length = hop_length

        self.K_val = int(
            torch.round(
                2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))
            ).item()
        )
        self.half_K = self.K_val // 2
        self.len_win = 2 * self.half_K + 1

        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.C = -1 / (2 * self.L**2)

        k = torch.arange(-self.half_K, self.half_K + 1, dtype=torch.float32)
        self.g = self.A * torch.exp(self.C * (k ** 2))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        self.N_input = x.shape[0]
        device = x.device

        g_v = self.g.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.half_K, self.half_K), mode="reflect")
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, x.shape[0], self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]
        tfr_segments = x_frames * g_v

        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t()

        m_vec = torch.arange(self.M, device=device).view(-1, 1)
        n_vec_row = n_vec.view(1, -1)

        phase_correction = torch.exp(
            1j * 2 * torch.pi / self.M * m_vec * (self.half_K - n_vec_row)
        )

        return tfr * phase_correction
    def rec(self, tfr):

        device = tfr.device
        M, num_frames = tfr.shape
        g_v = self.g.to(device)

        if not hasattr(self, "N_input"):
            raise ValueError("Call forward() before rec() so the original length is known.")

        n_vec = torch.arange(0, num_frames * self.hop_length, self.hop_length, device=device)[:num_frames]
        m_vec = torch.arange(M, device=device).view(-1, 1)
        phase_correction = torch.exp(
            1j * 2 * torch.pi / M * m_vec * (self.half_K - n_vec.view(1, -1))
        )
        tfr_origin = tfr / phase_correction

        tfr_segments = torch.fft.ifft(tfr_origin.t(), n=M, dim=1).real
        tfr_segments = tfr_segments[:, :self.len_win]

        if self.hop_length == 1:

            x_at_t = tfr_segments[:, self.half_K]

            h_0 = g_v[self.half_K]
            x_reconstructed = x_at_t / h_0

            return x_reconstructed[:self.N_input]

        else:
            len_padded_output = (num_frames - 1) * self.hop_length + self.len_win
            x_reconstructed = torch.zeros(len_padded_output, device=device, dtype=torch.float32)
            window_sum = torch.zeros(len_padded_output, device=device, dtype=torch.float32)

            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.len_win
                x_reconstructed[start:end] += tfr_segments[i] * g_v
                window_sum[start:end] += g_v ** 2

            mask = window_sum > self.eps
            x_reconstructed[mask] /= window_sum[mask]

            return x_reconstructed[self.half_K : self.half_K + self.N_input]
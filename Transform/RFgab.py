import torch
from Transform.Transformation import Transformation


class Rfgab(Transformation):
    """
    Reassigned Gabor time-frequency representation.
    """
    def __init__(self, M, hop_length=1, eps=1e-6, L=10, gamma_K=1e-4):
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
        k2 = k ** 2

        self.g = self.A * torch.exp(self.C * k2)
        self.dg = (self.L ** -2) * k * self.g
        self.tg = -k * self.g

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        device = x.device

        N = x.shape[0]

        g_v = self.g.to(device)
        dg_v = self.dg.to(device)
        tg_v = self.tg.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.half_K, self.half_K), mode="reflect")
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        num_frames = (N - 1) // self.hop_length + 1

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]

        tfr = torch.fft.fft(x_frames * g_v, n=self.M, dim=1).t()
        tfr_t = torch.fft.fft(x_frames * tg_v, n=self.M, dim=1).t()
        tfr_d = torch.fft.fft(x_frames * dg_v, n=self.M, dim=1).t()

        magnitude_sq = torch.abs(tfr) ** 2
        mask = torch.abs(tfr) > self.eps

        tau_m = torch.zeros_like(tfr, dtype=torch.float32)
        v_m = torch.zeros_like(tfr, dtype=torch.float32)

        tau_m[mask] = torch.real(tfr_t[mask] / tfr[mask])
        v_m[mask] = torch.imag(tfr_d[mask] / tfr[mask])

        col_indices = torch.arange(num_frames, device=device).view(1, -1).expand(self.M, num_frames)
        m_indices = torch.arange(self.M, device=device).view(-1, 1).expand(self.M, num_frames)

        n_hat = col_indices - torch.round(tau_m / self.hop_length).long()
        m_hat = m_indices + torch.round(v_m * self.M / (2 * torch.pi)).long()

        valid_bounds = (n_hat >= 0) & (n_hat < num_frames) & (m_hat >= 0) & (m_hat < self.M) & mask
        lost = torch.sum(magnitude_sq[mask & ~valid_bounds]).item()

        rtfr = torch.zeros((self.M, num_frames), dtype=torch.float32, device=device)

        indices_valides = torch.nonzero(valid_bounds, as_tuple=True)
        final_m_hat = m_hat[indices_valides]
        final_n_hat = n_hat[indices_valides]
        final_energy = magnitude_sq[indices_valides]

        rtfr.index_put_((final_m_hat, final_n_hat), final_energy, accumulate=True)

        return rtfr, lost
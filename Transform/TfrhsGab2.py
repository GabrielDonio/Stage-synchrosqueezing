import torch
from mmaxis import m_axis
from Transform.Transformation import Transformation


class Tfrthsgab2(Transformation):
    def __init__(
        self,
        M,
        eps=1e-6,
        L=10,
        gamma_K=1e-4,
        q_method=2,
        hop_length=1,
    ):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K
        self.q_method = q_method
        self.hop_length = hop_length

        self.K_val = int(
            torch.round(
                2
                * L
                * torch.sqrt(
                    torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K))
                )
            ).item()
        )
        self.half_K = self.K_val // 2
        self.len_win = 2 * self.half_K + 1

        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.C = -1 / (2 * self.L**2)

        k = torch.arange(-self.half_K, self.half_K + 1, dtype=torch.float32)
        k2 = k**2

        self.g = self.A * torch.exp(self.C * k2)
        self.tg = -k * self.g
        self.dg = -1 / (self.L**2) * self.tg
        self.tdg = -k2 / (self.L**2) * self.g
        self.t2g = k2 * self.g
        self.d2g = (-1 / (self.L**2) + k2 / (self.L**4)) * self.g

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.complex64).reshape(-1)
        device = x.device
        self.N_input = x.shape[0]

        g_v = self.g.to(device)
        tg_v = self.tg.to(device)
        dg_v = self.dg.to(device)
        tdg_v = self.tdg.to(device)
        t2g_v = self.t2g.to(device)
        d2g_v = self.d2g.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(
            x_3d, (self.half_K, self.half_K), mode="reflect"
        )
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, self.N_input, self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[
            :num_frames
        ]

        tfr_raw = torch.fft.fft(x_frames * g_v, n=self.M, dim=1).t()
        tfr_t_raw = torch.fft.fft(x_frames * tg_v, n=self.M, dim=1).t()
        tfr_d_raw = torch.fft.fft(x_frames * dg_v, n=self.M, dim=1).t()
        tfr_td_raw = torch.fft.fft(x_frames * tdg_v, n=self.M, dim=1).t()
        tfr_t2_raw = torch.fft.fft(x_frames * t2g_v, n=self.M, dim=1).t()
        tfr_d2_raw = torch.fft.fft(x_frames * d2g_v, n=self.M, dim=1).t()

        m_vec = torch.as_tensor(
            m_axis(self.M), device=device, dtype=torch.float32
        ).view(-1, 1)
        n_grid = n_vec.view(1, -1)
        phase_corr = torch.exp(-1j * 2 * torch.pi / self.M * m_vec * n_grid)
        tfr = tfr_raw * phase_corr

        mask = torch.abs(tfr_raw) > self.eps

        stfr = torch.zeros(
            (self.M, num_frames), dtype=torch.complex64, device=device
        )
        lost = torch.tensor(0.0, device=device, dtype=torch.float32)
        lost2 = torch.zeros((self.M, 1), dtype=torch.complex64, device=device)

        if torch.any(mask):
            tfr_v = tfr_raw[mask]
            tfr_t_v = tfr_t_raw[mask]
            tfr_d_v = tfr_d_raw[mask]
            tfr_td_v = tfr_td_raw[mask]
            tfr_t2_v = tfr_t2_raw[mask]
            tfr_d2_v = tfr_d2_raw[mask]
            tfr_phase_v = tfr[mask]

            m_indices, n_indices = torch.where(mask)
            n_float = n_indices.to(torch.float32)
            m_float = m_indices.to(torch.float32)

            R_t = tfr_t_v / tfr_v
            R_d = tfr_d_v / tfr_v

            n_tilde = n_float - R_t
            n_hat = n_float - torch.round(R_t.real)
            m_hat = m_float + torch.round(
                (self.M / (2 * torch.pi)) * R_d.imag
            )

            dt_hat_dt = tfr_td_v - tfr_t_v * tfr_d_v
            valid_dt = torch.abs(dt_hat_dt) > self.eps

            q_hat = torch.zeros_like(tfr_v, dtype=torch.complex64)

            if self.q_method == 2:
                denom = tfr_t_v * tfr_d_v - tfr_td_v * tfr_v
                valid_q = valid_dt & (torch.abs(denom) > self.eps)
                q_hat[valid_q] = (
                    tfr_d2_v[valid_q] * tfr_v[valid_q] - tfr_d_v[valid_q] ** 2
                ) / denom[valid_q]
            else:  
                denom = tfr_t_v**2 - tfr_t2_v * tfr_v
                valid_q = valid_dt & (torch.abs(denom) > self.eps)
                q_hat[valid_q] = (
                    tfr_td_v[valid_q] * tfr_v[valid_q]
                    + tfr_v[valid_q] ** 2
                    - tfr_t_v[valid_q] * tfr_d_v[valid_q]
                ) / denom[valid_q]

            q_abs = torch.abs(q_hat)
            q_imag = torch.imag(q_hat)

            fallback_cond = (
                (q_abs < self.eps)
                | (1.0 / (q_abs + 1e-12) < self.eps)
                | (torch.abs(q_imag) < self.eps)
            )

            term1 = torch.imag(q_hat * n_tilde) / (q_imag + 1e-12)
            term2 = torch.round(
                (2 * torch.pi / self.M)
                / (q_imag + 1e-12)
                * (m_float - m_hat)
            )

            n_hat_q_calc = torch.round(term1 + term2)
            n_hat_q_calc[torch.isnan(n_hat_q_calc)] = n_hat[
                torch.isnan(n_hat_q_calc)
            ]

            n_hat_q = torch.where(fallback_cond, n_hat, n_hat_q_calc)
            n_hat_q_int = n_hat_q.to(torch.long)

            n_out_of_bounds = (n_hat_q_int < 0) | (n_hat_q_int >= num_frames)
            in_bounds = ~n_out_of_bounds

            lost = torch.sum(torch.abs(tfr_v[n_out_of_bounds]) ** 2)

            if torch.any(n_out_of_bounds):
                m_out = m_indices[n_out_of_bounds]
                tfr_out = tfr_phase_v[n_out_of_bounds]
                lost2.index_add_(0, m_out, tfr_out.unsqueeze(1))

            if torch.any(in_bounds):
                m_targets = m_indices[in_bounds]
                n_targets = n_hat_q_int[in_bounds]
                tfr_vals = tfr_phase_v[in_bounds]

                flat_targets = m_targets * num_frames + n_targets
                stfr_flat = torch.zeros(
                    self.M * num_frames, dtype=torch.complex64, device=device
                )
                stfr_flat.index_add_(0, flat_targets, tfr_vals)
                stfr = stfr_flat.view(self.M, num_frames)

        return stfr, (lost, lost2)
    def rec(self, stfr):
        if not hasattr(self, "N_input"):
            raise ValueError(
                "N_input n'est pas défini. Veuillez exécuter forward() avant rec()."
            )
        
        device = stfr.device
        M = self.M
        N = self.N_input
        S = torch.sum(stfr, dim=1)

        s_hat_M = torch.fft.ifft(S, n=M)

        if N == M:
            s_hat = s_hat_M
        else:
            indices = torch.arange(N, device=device) % M
            s_hat = s_hat_M[indices]
        return s_hat.real
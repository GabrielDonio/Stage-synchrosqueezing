import torch
import torch.nn as nn
from Transform.Transformation import Transformation


def m_axis(M):
    return torch.arange(M, dtype=torch.float32)


class Tfrvsgab2(Transformation):

    def __init__(
        self,
        M,
        eps=1e-6,
        L=10.0,
        q_method=2,
        if_method=1,
        gamma_K=1e-4,
        q_threshold=1e-4,
        hop_length=1,
    ):
        super().__init__(M)
        self.L = float(L)
        self.q_method = q_method
        self.if_method = if_method
        self.gamma_K = gamma_K
        self.q_threshold = q_threshold
        self.hop_length = hop_length

        K_val = int(
            torch.round(
                2
                * self.L
                * torch.sqrt(
                    torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K))
                )
            ).item()
        )
        self.half_K = K_val // 2
        self.len_win = 2 * self.half_K + 1

        k = torch.arange(
            -self.half_K, self.half_K + 1, dtype=torch.float32
        )
        k2 = k**2

        A = 1.0 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        C = -1.0 / (2.0 * (self.L**2))

        g = A * torch.exp(C * k2)
        tg = -k * g
        dg = -(1.0 / (self.L**2)) * tg
        tdg = -(k2 / (self.L**2)) * g
        t2g = k2 * g
        d2g = (-(1.0 / (self.L**2)) + (k2 / (self.L**4))) * g
        t3g = -k * t2g

        self.register_buffer("g", g)
        self.register_buffer("tg", tg)
        self.register_buffer("dg", dg)
        self.register_buffer("tdg", tdg)
        self.register_buffer("t2g", t2g)
        self.register_buffer("d2g", d2g)
        self.register_buffer("t3g", t3g)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.complex64).reshape(-1)
        device = x.device
        N_input = x.shape[0]
        self.N_input = N_input

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(
            x_3d, (self.half_K, self.half_K), mode="reflect"
        )
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, N_input, self.hop_length, device=device)
        num_frames = n_vec.shape[0]


        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[
            :num_frames
        ]

        tfr_raw = torch.fft.fft(x_frames * self.g, n=self.M, dim=1).t()
        tfr_t_raw = torch.fft.fft(x_frames * self.tg, n=self.M, dim=1).t()
        tfr_d_raw = torch.fft.fft(x_frames * self.dg, n=self.M, dim=1).t()
        tfr_td_raw = torch.fft.fft(x_frames * self.tdg, n=self.M, dim=1).t()
        tfr_t2_raw = torch.fft.fft(x_frames * self.t2g, n=self.M, dim=1).t()
        tfr_d2_raw = torch.fft.fft(x_frames * self.d2g, n=self.M, dim=1).t()

        if self.q_method == 4:
            tfr_t3_raw = torch.fft.fft(x_frames * self.t3g, n=self.M, dim=1).t()

        m_vec = m_axis(self.M).to(device).view(-1, 1)
        n_grid = n_vec.view(1, -1)
        phase_corr = torch.exp(-1j * 2 * torch.pi / self.M * m_vec * n_grid)
        tfr = tfr_raw * phase_corr

        mask = torch.abs(tfr_raw) > self.eps

        stfr = torch.zeros(
            (self.M, num_frames), dtype=torch.complex64, device=device
        )
        q_hatmap = torch.zeros(
            (self.M, num_frames), dtype=torch.complex64, device=device
        )
        if_hatmap = torch.zeros(
            (self.M, num_frames), dtype=torch.float32, device=device
        )
        lost = torch.tensor(0.0, device=device, dtype=torch.float32)

        if torch.any(mask):
            tfr_v = tfr_raw[mask]
            tfr_t_v = tfr_t_raw[mask]
            tfr_d_v = tfr_d_raw[mask]
            tfr_td_v = tfr_td_raw[mask]
            tfr_t2_v = tfr_t2_raw[mask]
            tfr_d2_v = tfr_d2_raw[mask]

            R_t = tfr_t_v / tfr_v
            R_d = tfr_d_v / tfr_v

            n_displ_round = torch.complex(
                torch.round(R_t.real), torch.round(R_t.imag)
            )
            m_indices, n_indices = torch.where(mask)

            
            if self.q_method == 1:  
                alpha_hat_denum = torch.imag(tfr_t_v * torch.conj(tfr_v))
                valid_q = torch.abs(alpha_hat_denum) > self.q_threshold
                alpha_hat = torch.zeros_like(alpha_hat_denum)
                alpha_hat[valid_q] = (
                    torch.real(tfr_d_v[valid_q] * torch.conj(tfr_v[valid_q]))
                    / alpha_hat_denum[valid_q]
                )
                q_hat = 1j * alpha_hat

            elif self.q_method == 2:  
                q_hat_denum = tfr_t_v * tfr_d_v - tfr_td_v * tfr_v
                valid_q = torch.abs(q_hat_denum) > self.q_threshold
                q_hat = torch.zeros_like(q_hat_denum, dtype=torch.complex64)
                q_hat[valid_q] = (
                    tfr_d2_v[valid_q] * tfr_v[valid_q] - tfr_d_v[valid_q] ** 2
                ) / q_hat_denum[valid_q]
                alpha_hat = q_hat.imag

            elif self.q_method == 3: 
                q_hat_denum = tfr_t_v**2 - tfr_t2_v * tfr_v
                valid_q = torch.abs(q_hat_denum) > self.q_threshold
                q_hat = torch.zeros_like(q_hat_denum, dtype=torch.complex64)
                q_hat[valid_q] = (
                    tfr_td_v[valid_q] * tfr_v[valid_q]
                    - tfr_t_v[valid_q] * tfr_d_v[valid_q]
                    + tfr_v[valid_q] ** 2
                ) / q_hat_denum[valid_q]
                alpha_hat = q_hat.imag

            elif self.q_method == 4:  
                tfr_t3_v = tfr_t3_raw[mask]
                tfr_dt2_v = 2 * tfr_t_v + tfr_t2_v

                Ar = torch.stack(
                    [
                        torch.stack([tfr_t2_v, -tfr_t_v, tfr_v], dim=-1),
                        torch.stack([tfr_dt2_v, -tfr_td_v, tfr_d_v], dim=-1),
                        torch.stack([tfr_t3_v, -tfr_t2_v, tfr_t_v], dim=-1),
                    ],
                    dim=-2,
                )
                ur = torch.stack(
                    [tfr_d_v, tfr_d2_v, tfr_td_v + tfr_v], dim=-1
                ).unsqueeze(-1)

                q_hat_denum = torch.linalg.det(Ar)
                valid_q = torch.abs(q_hat_denum) > self.q_threshold
                q_hat = torch.zeros_like(q_hat_denum, dtype=torch.complex64)

                if torch.any(valid_q):
                    Ar_inv = torch.linalg.pinv(Ar[valid_q])
                    ur_val = ur[valid_q]
                    rx = torch.matmul(Ar_inv[:, 0:1, :], ur_val).squeeze(-1)
                    q_hat_part = torch.matmul(
                        Ar_inv[:, 1:2, :], ur_val
                    ).squeeze(-1) - 2 * rx * n_displ_round[valid_q].unsqueeze(
                        -1
                    )
                    q_hat_part = q_hat_part.squeeze(-1)

                    bound_mask = torch.abs(q_hat_part) <= (
                        1.0 / self.q_threshold
                    )
                    q_hat_valid = torch.zeros_like(q_hat_part)
                    q_hat_valid[bound_mask] = q_hat_part[bound_mask]
                    q_hat[valid_q] = q_hat_valid

                alpha_hat = q_hat.imag

            q_hatmap[m_indices, n_indices] = q_hat

            m_hat_base = m_indices.to(torch.float32) + (
                self.M / (2 * torch.pi)
            ) * torch.imag(R_d)

            if self.if_method == 1:
                m_hat_q = torch.round(m_hat_base) + torch.round(
                    (self.M / (2 * torch.pi))
                    * alpha_hat
                    * torch.round(torch.real(R_t))
                )
                if_val = (
                    m_hat_base
                    + (self.M / (2 * torch.pi))
                    * alpha_hat
                    * torch.real(R_t)
                ) / self.M

            elif self.if_method == 2:
                m_hat_q = torch.round(m_hat_base) + torch.round(
                    (self.M / (2 * torch.pi))
                    * torch.imag(q_hat * n_displ_round)
                )
                if_val = (
                    m_hat_base
                    + (self.M / (2 * torch.pi)) * torch.imag(q_hat * R_t)
                ) / self.M

            elif self.if_method == 3: 
                m_hat_q = torch.round(m_hat_base)
                if_val = m_hat_base / self.M

            m_hat_q_int = m_hat_q.to(torch.long)
            if_hatmap[m_indices, n_indices] = if_val

            out_of_bounds = (m_hat_q_int < 0) | (m_hat_q_int >= self.M)
            in_bounds = ~out_of_bounds

            lost = torch.sum(torch.abs(tfr_v[out_of_bounds]) ** 2)

            if torch.any(in_bounds):
                m_targets = m_hat_q_int[in_bounds] % self.M  
                n_targets = n_indices[in_bounds]
                tfr_vals = tfr_v[in_bounds]

                flat_targets = m_targets * num_frames + n_targets
                stfr_flat = torch.zeros(
                    self.M * num_frames, dtype=torch.complex64, device=device
                )
                stfr_flat.index_add_(
                    0, flat_targets, tfr_vals / (2 * torch.pi)
                )
                stfr = stfr_flat.view(self.M, num_frames)

        return tfr, stfr, lost, q_hatmap, if_hatmap
    def rec(self, stfr):
        if self.hop_length != 1:
            raise ValueError(
                "La reconstruction ponctuelle (rec_mor) nécessite hop_length = 1."
            )

        if not hasattr(self, "N_input"):
            raise ValueError(
                "N_input n'est pas défini"
            )

        device = stfr.device
        M, num_frames = stfr.shape
        g_v = self.g.to(device)

        stfr_corrected = stfr * (2 * torch.pi)
        tfr_segments = torch.fft.ifft(stfr_corrected.t(), n=M, dim=1)
        
        tfr_segments = tfr_segments[:, : self.len_win]

        x_at_t = tfr_segments[:, self.half_K]
        h_0 = g_v[self.half_K]

        if torch.abs(h_0) < self.eps:
            raise ValueError(
                "Le centre de la fenêtre de Gabor est nul ou trop petit."
            )

        x_reconstructed = x_at_t / h_0

        return x_reconstructed[: self.N_input].real
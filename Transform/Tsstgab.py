import torch
from mmaxis import m_axis
from Transform.Transformation import Transformation


class TSST_gab(Transformation):

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
        self.g = self.A * torch.exp(self.C * (k ** 2))
        self.tg = -k * self.g  

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.complex64, device=x.device).reshape(-1)
        self.N_input = x.shape[0] 
        device = x.device

        g_v = self.g.to(device)
        tg_v = self.tg.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.half_K, self.half_K), mode="reflect")
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, self.N_input, self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]

        tfr_fft = torch.fft.fft(x_frames * g_v, n=self.M, dim=1).t()
        tfr_t_fft = torch.fft.fft(x_frames * tg_v, n=self.M, dim=1).t()

        m_vec = m_axis(self.M, device=device).view(-1, 1)
        fft_phase_correction = torch.exp(2j * torch.pi * m_vec * self.half_K / self.M)

        tfr_base = tfr_fft * fft_phase_correction
        tfr_t_base = tfr_t_fft * fft_phase_correction

        m_grid = torch.arange(self.M, device=device).view(-1, 1)
        n_grid = torch.arange(num_frames, device=device).view(1, -1)
        global_phase_corr = torch.exp(-2j * torch.pi * m_grid * (n_grid * self.hop_length) / self.M)
        
        tfr_base = tfr_base * global_phase_corr
        tfr_t_base = tfr_t_base * global_phase_corr

        magnitude = torch.abs(tfr_base)
        mask = magnitude > self.eps

        tau_m = torch.zeros_like(tfr_base, dtype=torch.float32)
        tau_m[mask] = torch.real(tfr_t_base[mask] / tfr_base[mask])

        col_indices = torch.arange(num_frames, device=device).view(1, -1).expand(self.M, num_frames)
        n_hat = col_indices - torch.round(tau_m / self.hop_length).long()

        valid_bounds = (n_hat >= 0) & (n_hat < num_frames) & mask
        lost_mask = mask & (~valid_bounds)
        lost = torch.sum(magnitude[lost_mask] ** 2).item()

        rtfr = torch.zeros((self.M, num_frames), dtype=torch.complex64, device=device)

        indices_valides = torch.nonzero(valid_bounds, as_tuple=True)
        m_dest = indices_valides[0]        
        n_dest = n_hat[indices_valides]    # réallocation du temps
        
        flat_dest_indices = m_dest * num_frames + n_dest
        
        flat_valeurs = tfr_base[indices_valides] / (2 * torch.pi)

        rtfr.view(-1).index_add_(0, flat_dest_indices, flat_valeurs)

        return rtfr, lost

    def rec(self, rtfr):
        device = rtfr.device
        M, num_frames = rtfr.shape

        if self.hop_length != 1:
            print("On ne peut pas reconstruire avec hop length > 1")
            return None
        m_grid = torch.arange(M, device=device).view(-1, 1)
        n_grid = torch.arange(num_frames, device=device).view(1, -1)
        
        inverse_global_phase = torch.exp(2j * torch.pi * m_grid * (n_grid * self.hop_length) / M)
        rtfr_phased = rtfr * inverse_global_phase
        x_reconstructed = torch.sum(rtfr_phased, dim=0).real

        h_0 = torch.sum(self.g.to(device)).real

        scale = (2 * torch.pi) / (M * h_0)
        
        x_final = x_reconstructed * scale

        if hasattr(self, 'N_input'):
            if x_final.shape[0] < self.N_input:
                padding = torch.zeros(self.N_input - x_final.shape[0], device=device)
                x_final = torch.cat([x_final, padding])
            else:
                x_final = x_final[:self.N_input]

        return x_final
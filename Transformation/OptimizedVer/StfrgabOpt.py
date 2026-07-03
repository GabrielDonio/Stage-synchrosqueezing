import torch
from Transform.Transformation import Transformation

class Stfrgab(Transformation):
    """
    Implementation of the synchrosqueezing transform based on the Gabor transform.
    """
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K

        self.K_val = int(torch.round(2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))).item())
        self.half_K = self.K_val // 2
        self.len_win = 2 * self.half_K + 1

        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.C = -1 / (2 * self.L**2)

        k = torch.arange(-self.half_K, self.half_K + 1, dtype=torch.float32)
        self.g = self.A * torch.exp(self.C * (k ** 2))       # Fenêtre de Gabor
        self.dg = (self.L ** -2) * k * self.g                # Dérivée analytique de Gabor

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.complex64, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        g_v = self.g.to(device)
        dg_v = self.dg.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.half_K, self.half_K), mode='reflect')
        x_padded = x_padded_3d.squeeze(0).squeeze(0)
        
        x_frames = x_padded.unfold(0, self.len_win, 1)[:N]

        tfr_segments = x_frames * g_v
        tfr_d_segments = x_frames * dg_v

        tfr_fft = torch.fft.fft(tfr_segments, n=self.M, dim=1).t()    # (M, N)
        tfr_d_fft = torch.fft.fft(tfr_d_segments, n=self.M, dim=1).t() # (M, N)

        m_vec = torch.arange(self.M, device=device).view(-1, 1)
        fft_phase_correction = torch.exp(2j * torch.pi * m_vec * self.half_K / self.M)
        
        tfr_base = tfr_fft * fft_phase_correction
        tfr_d_base = tfr_d_fft * fft_phase_correction

        magnitude = torch.abs(tfr_base)
        mask = magnitude > self.eps

        v_m = torch.zeros_like(tfr_base, dtype=torch.float32)
        v_m[mask] = torch.imag(tfr_d_base[mask] / tfr_base[mask])

        m_indices = torch.arange(self.M, device=device).view(-1, 1).expand(self.M, N)
        m_hat = m_indices + torch.round(v_m * self.M / (2 * torch.pi)).long()
        
        valid_bounds = (m_hat >= 0) & (m_hat < self.M) & mask
        lost_mask = mask & (~valid_bounds)
        lost = torch.sum(magnitude[lost_mask] ** 2).item()
        
        rtfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)
        
        indices_valides = torch.nonzero(valid_bounds, as_tuple=True)
        m_dest = m_hat[indices_valides]
        n_dest = indices_valides[1]
        
        flat_dest_indices = m_dest * N + n_dest

        flat_valeurs = tfr_base[indices_valides] / (2 * torch.pi)
        
        rtfr.view(-1).index_add_(0, flat_dest_indices, flat_valeurs)
                    
        return rtfr, lost
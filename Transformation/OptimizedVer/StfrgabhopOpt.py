import torch
from Transformation.Transformation import Transformation

class Stfrgabhop(Transformation):
    """
    Implementation of the Synchrosqueezing transform based on the Gabor transform
    with support for hop_length.
    """
    def __init__(self, M, hop_length, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K
        self.hop_length = hop_length

        self.K_val = int(torch.round(2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))).item())
        self.half_K = self.K_val // 2
        self.len_win = 2 * self.half_K + 1

        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.C = -1 / (2 * self.L**2)

        k = torch.arange(-self.half_K, self.half_K + 1, dtype=torch.float32)
        self.g = self.A * torch.exp(self.C * (k ** 2))       # Gabor
        self.dg = (self.L ** -2) * k * self.g                # Dérivée de Gabor

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        g_v = self.g.to(device)
        dg_v = self.dg.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.half_K, self.half_K), mode='reflect')
        x_padded = x_padded_3d.squeeze(0).squeeze(0)
        
        n_vec = torch.arange(0, N, self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]

        tfr_fft = torch.fft.fft(x_frames * g_v, n=self.M, dim=1).t()   
        tfr_d_fft = torch.fft.fft(x_frames * dg_v, n=self.M, dim=1).t()

        m_vec = torch.arange(self.M, device=device).view(-1, 1)
        fft_phase_correction = torch.exp(2j * torch.pi * m_vec * self.half_K / self.M)
        
        tfr_base = tfr_fft * fft_phase_correction
        tfr_d_base = tfr_d_fft * fft_phase_correction
        
        magnitude = torch.abs(tfr_base)
        mask = magnitude > self.eps

        v_m = torch.zeros_like(tfr_base, dtype=torch.float32)
        v_m[mask] = torch.imag(tfr_d_base[mask] / tfr_base[mask])

        m_indices = torch.arange(self.M, device=device).view(-1, 1).expand(self.M, num_frames)
        m_hat = m_indices + torch.round(v_m * self.M / (2 * torch.pi)).long()
        
        valid_bounds = (m_hat >= 0) & (m_hat < self.M) & mask
        lost_mask = mask & (~valid_bounds)
        lost = torch.sum(magnitude[lost_mask] ** 2).item()

        rtfr = torch.zeros((self.M, num_frames), dtype=torch.complex64, device=device)
        
        indices_valides = torch.nonzero(valid_bounds, as_tuple=True)
        m_dest = m_hat[indices_valides]
        n_dest = indices_valides[1] 
        flat_dest_indices = m_dest * num_frames + n_dest
        flat_valeurs = tfr_base[indices_valides] / (2 * torch.pi)
        
        rtfr.view(-1).index_add_(0, flat_dest_indices, flat_valeurs)
                    
        return rtfr, lost
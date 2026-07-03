import torch
from Transform.Transformation import Transformation

class Stfrwin(Transformation):
    def __init__(self, M, window, eps=0.01):
        super().__init__(M, eps)
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32)
        self.len_win = self.window.shape[0]
        self.dw = -torch.gradient(self.window)[0]

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        w_v = self.window.to(device)
        dw_v = self.dw.to(device)
        
        pad_amount = self.len_win // 2
        x_padded = torch.nn.functional.pad(x, (pad_amount, pad_amount), mode='constant', value=0)
        
        x_frames = x_padded.unfold(0, self.len_win, 1)[:N]
        
        tfr_segments = x_frames * w_v
        tfr_d_segments = x_frames * dw_v
        
        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t()    
        tfr_d = torch.fft.fft(tfr_d_segments, n=self.M, dim=1).t() 
        
        m_vec = torch.arange(self.M, device=device).view(-1, 1)
        fft_phase_correction = torch.exp(2j * torch.pi * m_vec * pad_amount / self.M)
        tfr = tfr * fft_phase_correction
        tfr_d = tfr_d * fft_phase_correction
        
        magnitude = torch.abs(tfr)
        max_per_col, _ = torch.max(magnitude, dim=0, keepdim=True)
        mask = magnitude > (self.eps * max_per_col)
        
        v_m = torch.zeros_like(tfr, dtype=torch.float32)
        v_m[mask] = torch.imag(tfr_d[mask] / tfr[mask])
        
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
        flat_valeurs = tfr[indices_valides] 
        
        rtfr.view(-1).index_add_(0, flat_dest_indices, flat_valeurs)
        
        return rtfr, lost
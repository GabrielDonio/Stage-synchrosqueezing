import torch
from Transformation.Transformation import Transformation
from mmaxis import m_axis

class Stfrwin(Transformation):
    def __init__(self, M, window, eps=1e-6):
        super().__init__(M, eps)

        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32)
        self.len_win = self.window.shape[0]

        self.dw = -torch.gradient(self.window)[0] 
        self.B = -1j * 2 * torch.pi / self.M

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        w_v = self.window.to(device)
        dw_v = self.dw.to(device)

        rtfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device) 
        mm = m_axis(self.M, device=device).view(-1, 1)

        center = self.len_win // 2
        left = center
        right = self.len_win - center - 1
        
        lost = 0.0
        
        for n in range(N):
            k_min = min(n, left)
            k_max = min(N - 1 - n, right)

            k = torch.arange(-k_min, k_max + 1, device=device)
            
            w_slice = w_v[center - k_min : center + k_max + 1]
            dg_slice = dw_v[center - k_min : center + k_max + 1]
            x_slice = x[n + k]

            exp_B_mm_k = torch.exp(self.B * mm * k)
            exp_B_mm_n = torch.exp(self.B * mm * n)

            tfr_n = exp_B_mm_n.squeeze() * torch.sum(x_slice * w_slice * exp_B_mm_k, dim=1)
            tfr_d_n = exp_B_mm_n.squeeze() * torch.sum(x_slice * dg_slice * exp_B_mm_k, dim=1)

            magnitude = torch.abs(tfr_n)
            mask = magnitude > self.eps
            
            if not torch.any(mask):
                continue

            m_indices = torch.arange(self.M, device=device)[mask]
            v_m = torch.imag(tfr_d_n[mask] / tfr_n[mask])
            m_hat = m_indices + torch.round(v_m * self.M / (2 * torch.pi)).long()

            valid_bounds = (m_hat >= 0) & (m_hat < self.M)
            
            lost += torch.sum(magnitude[mask][~valid_bounds] ** 2).item()

            final_m_hat = m_hat[valid_bounds]
            final_m_orig = m_indices[valid_bounds]

            phase_correction = torch.exp(2 * 1j * torch.pi * final_m_orig * n / self.M)
            valeurs = tfr_n[final_m_orig] * phase_correction

            rtfr[:, n].index_add_(0, final_m_hat, valeurs)
                    
        return rtfr, lost
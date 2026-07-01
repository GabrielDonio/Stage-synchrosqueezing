import torch
from mmaxis import m_axis
from Transformation.Transformation import Transformation

class Tfrrgab(Transformation):
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K

        # Précalcul de K une fois pour toutes (évite les calculs d'entiers dans forward)
        self.K_val = int(torch.round(2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))).item())
        self.half_K = self.K_val // 2
        
        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.B = -1j * 2 * torch.pi / self.M
        self.C = -1 / (2 * self.L**2)
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device

        rtfr = torch.zeros((self.M, N), dtype=torch.float32, device=device) 
        lost = 0.0

        mm = m_axis(self.M, device=device).view(-1, 1)
        
        for n in range(N):
            k_min = min(n, self.half_K)
            k_max = min(N - 1 - n, self.half_K) 

            k = torch.arange(-k_min, k_max + 1, device=device)
            k2 = k ** 2

            g = self.A * torch.exp(self.C * k2)
            dg = (self.L ** -2) * k * g
            tg = -k * g 

            x_slice = x[n + k]

            exp_B_mm_k = torch.exp(self.B * mm * k)
            exp_B_mm_n = torch.exp(self.B * mm * n)
            
            common_phase = exp_B_mm_n.squeeze()
            tfr_n = common_phase * torch.sum(x_slice * g * exp_B_mm_k, dim=1)
            tfr_t_n = common_phase * torch.sum(x_slice * tg * exp_B_mm_k, dim=1)
            tfr_d_n = common_phase * torch.sum(x_slice * dg * exp_B_mm_k, dim=1)
                
            magnitude_sq = torch.abs(tfr_n) ** 2
            mask = torch.abs(tfr_n) > self.eps
            
            if not torch.any(mask):
                continue
            
            m_indices = torch.arange(self.M, device=device)[mask]
            
            tau_m = torch.real(tfr_t_n[mask] / tfr_n[mask])
            n_hat = n - torch.round(tau_m).long()
            
            v_m = torch.imag(tfr_d_n[mask] / tfr_n[mask])
            m_hat = m_indices + torch.round(v_m * self.M / (2 * torch.pi)).long()
            
            valid_bounds = (n_hat >= 0) & (n_hat < N) & (m_hat >= 0) & (m_hat < self.M)
            
            lost += torch.sum(magnitude_sq[mask][~valid_bounds]).item()
            
            final_m_hat = m_hat[valid_bounds]
            final_n_hat = n_hat[valid_bounds]
            final_energy = magnitude_sq[mask][valid_bounds]

            rtfr.index_put_((final_m_hat, final_n_hat), final_energy, accumulate=True)
                    
        return rtfr, lost
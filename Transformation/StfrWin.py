import torch
from Transform.Transformation import Transformation
from mmaxis import m_axis

class Stfrwin(Transformation):
    def __init__(self, M, window, eps=1e-6):
        super().__init__(M, eps)
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32)
        self.len_win = self.window.shape[0]

        # Dérivée de la fenêtre 
        self.dw = -torch.gradient(self.window)[0] 
        self.B = -1j * 2 * torch.pi / self.M

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        w_v = self.window.to(device)
        dw_v = self.dw.to(device)

        rtfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device) 
        mm = m_axis(self.M, device=device)

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

            for m in range(self.M):
                exp_B_mm_k = torch.exp(self.B * mm[m] * k)
                exp_B_mm_n = torch.exp(self.B * mm[m] * n)
                

                tfr_m_n = exp_B_mm_n * torch.sum(x_slice * w_slice * exp_B_mm_k)
                
                if torch.abs(tfr_m_n) > self.eps:
                    tfr_d_m_n = exp_B_mm_n * torch.sum(x_slice * dg_slice * exp_B_mm_k)

                    m_hat = m + int(torch.round((self.M / (2 * torch.pi)) * torch.imag(tfr_d_m_n / tfr_m_n)).item())
 
                    if (m_hat < 0) or (m_hat >= self.M):
                        lost += torch.abs(tfr_m_n).item() ** 2
                        continue

                    val = tfr_m_n * torch.exp(2 * 1j * torch.pi * mm[m] * n / self.M)
                    rtfr[m_hat, n] = rtfr[m_hat, n] + val
                    
        return rtfr, lost
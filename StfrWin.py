import torch
from Transformation import Transformation
from mmaxis import m_axis

class Stfrwin(Transformation):
    def __init__(self, M, window, eps=1e-6):
        super().__init__(M, eps)

        self.window = torch.as_tensor(window, dtype=torch.float32)
        self.len_win = self.window.shape[0]

        # Dérivée de la fenêtre (vitesse)
        self.dw = -torch.gradient(self.window)[0] 
        self.B = -1j * 2 * torch.pi / self.M

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        w_v = self.window.to(device)
        dw_v = self.dw.to(device)

        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)
        tfr_d = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

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
            nn = n

            for m in range(self.M):
                exp_B_mm_k = torch.exp(self.B * mm[m] * k)
                exp_B_mm_nn = torch.exp(self.B * mm[m] * nn)
                
                tfr[m, n] = exp_B_mm_nn * torch.sum(x_slice * w_slice * exp_B_mm_k)
                
                if torch.abs(tfr[m, n]) > self.eps:

                    tfr_d[m, n] = exp_B_mm_nn * torch.sum(x_slice * dg_slice * exp_B_mm_k)

                    m_hat = m + int(torch.round((self.M / (2 * torch.pi)) * torch.imag(tfr_d[m, n] / tfr[m, n])).item())
 
                    m_out_of_bounds = (m_hat < 0) or (m_hat >= self.M)
                    
                    if m_out_of_bounds:
                        lost += torch.abs(tfr[m, n]).item() ** 2
                        continue

                    val = (tfr[m, n]) * torch.exp(2 * 1j * torch.pi * mm[m] * nn / self.M)
                    rtfr[m_hat, n] = rtfr[m_hat, n] + val
                    
        return rtfr, lost
import torch
import matplotlib.pyplot as plt
class Stfrgab(torch.nn.Module):
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__()

        self.M = M
        self.L = L
        self.gamma_K = gamma_K
        self.eps = eps

        self.K = 2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))
        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.B = -1j * 2 * torch.pi / self.M
        self.C = -1 / (2 * self.L**2)
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        lost = 0.0
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)
        rtfr = torch.zeros((self.M, N), dtype=torch.float32, device=device) 
        
        tfr_d = torch.zeros((self.M, N), dtype=torch.complex64, device=device)
        

        mm = torch.arange(0, self.M, device=device)
        
        for n in range(N):
            k_min = min(n, int(torch.round(self.K / 2).item()))
            k_max = min(N - 1 - n, int(torch.round(self.K / 2).item()))

            k = torch.arange(-k_min, k_max + 1, device=device)
            k2 = k ** 2

            g = self.A * torch.exp(self.C * k2) # fenetre de gabor
            dg = (self.L ** -2) * k * g #derivée de la fenetre de gabor

            nn = n

            for m in range(self.M):
                exp_B_mm_k = torch.exp(self.B * mm[m] * k)
                exp_B_mm_nn = torch.exp(self.B * mm[m] * nn)
                
                x_slice = x[n + k]
                
                tfr[m, n] = exp_B_mm_nn * torch.sum(x_slice * g * exp_B_mm_k)
                
                if torch.abs(tfr[m, n]) > self.eps:
                    tfr_d[m, n] = exp_B_mm_nn * torch.sum(x_slice * dg * exp_B_mm_k)
                    
                    m_hat = m + int(torch.round((self.M / (2 * torch.pi)) * torch.imag(tfr_d[m, n] / tfr[m, n])).item()) # seulement réalocation en fréquence
 
                    m_out_of_bounds = (m_hat < 0) or (m_hat >= self.M)
                    
                    if  m_out_of_bounds:
                        lost += torch.abs(tfr[m, n]).item() ** 2
                        continue
                    
                    rtfr[m_hat, n] = rtfr[m_hat, n] + tfr[m, n]/(2*torch.pi) * torch.exp(2*1j*torch.pi*mm[m]*nn/M)
                    
        return rtfr, lost
if __name__ == "__main__":
    M = 512
    tfrgab = Stfrgab(M)
    fs = 100
    f = 50
    t = torch.arange(0, 1, 1/fs)
    x = torch.cos(2 * torch.pi * f * t**2)
    rtfr, lost = tfrgab(x)
    print(lost)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Synchrosqueezed (Gabor)")
    plt.colorbar()
    plt.show()
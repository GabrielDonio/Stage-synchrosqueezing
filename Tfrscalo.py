import torch
import numpy as np
import matplotlib.pyplot as plt
class Tfrscalo(torch.nn.Module):
    def __init__(self, M, T, Ts, w0, eps=1e-6, gamma_K=1e-5, as_range=[0.05, 1.5]):
        super().__init__()

        self.M = M
        self.T = T          
        self.Ts = Ts        
        self.w0 = w0        
        self.gamma_K = gamma_K
        self.eps = eps
        
        self.scales = torch.tensor(np.logspace(np.log10(as_range[0]), np.log10(as_range[1]), M))

        self.sqrt_pi = torch.sqrt(torch.tensor(torch.pi))
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        sqrt_log_gamma = torch.sqrt(2 * torch.log(torch.tensor(1.0 / self.gamma_K)))

        for n in range(N):
            for m in range(self.M):
                a = self.scales[m]
                
                K = int(torch.round(sqrt_log_gamma * (self.T / self.Ts) * a).item())
                
                k_min = min(n, K)
                k_max = min(N - 1 - n, K)
                
                k = torch.arange(-k_min, k_max + 1, device=device)
                tau = k * self.Ts
                
                x_slice = x[n + k]
                
                gauss = torch.exp(- (tau ** 2) / (2 * (self.T * a) ** 2))
                
                osc = torch.exp(-1j * self.w0 * tau / a)
                
                norm = self.Ts / torch.sqrt(torch.abs(a) * self.T * self.sqrt_pi)
                
                tfr[m, n] = norm * torch.sum(x_slice * gauss * osc)
                
        return tfr

if __name__ == "__main__":
    M = 64
    T = 1.0
    Ts = 0.01
    f = 50
    w0 = 5.0  
    
    scalo = Tfrscalo(M=M, T=T, Ts=Ts, w0=w0)
    x = torch.cos(2*torch.pi*f*torch.linspace(0, 10, 500)**2) 
    
    tfr = scalo(x)
    plt.imshow(torch.log1p(tfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Scalogram)")
    plt.colorbar()
    plt.show() 
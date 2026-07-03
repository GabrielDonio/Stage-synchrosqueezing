import torch
from mmaxis import m_axis
from Transform.Transformation import Transformation
class Tfrrgab(Transformation):
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.L = L
        self.gamma_K = gamma_K

        self.K = 2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K))) #seuillage pour la fenetre de gabor
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
        
        tfr_t = torch.zeros((self.M, N), dtype=torch.complex64, device=device)
        tfr_d = torch.zeros((self.M, N), dtype=torch.complex64, device=device)
        

        mm = m_axis(self.M, device=device)
        
        for n in range(N):
            k_min = min(n, int(torch.round(self.K / 2).item()))
            k_max = min(N - 1 - n, int(torch.round(self.K / 2).item())) 

            k = torch.arange(-k_min, k_max + 1, device=device)
            k2 = k ** 2

            g = self.A * torch.exp(self.C * k2) # fenetre de gabor
            dg = (self.L ** -2) * k * g #derivée de la fenetre de gabor
            tg = -k * g 

            nn = n

            for m in range(self.M):
                exp_B_mm_k = torch.exp(self.B * mm[m] * k)
                exp_B_mm_nn = torch.exp(self.B * mm[m] * nn)
                
                x_slice = x[n + k]
                
                tfr[m, n] = exp_B_mm_nn * torch.sum(x_slice * g * exp_B_mm_k)
                
                if torch.abs(tfr[m, n]) > self.eps:

                    tfr_t[m, n] = exp_B_mm_nn * torch.sum(x_slice * tg * exp_B_mm_k) 
                    tfr_d[m, n] = exp_B_mm_nn * torch.sum(x_slice * dg * exp_B_mm_k) #STFT avec la fenetre dérivée
                    
                    n_hat = n - int(torch.round(torch.real(tfr_t[m, n] / tfr[m, n])).item()) #realocation en temps
                    m_hat = m + int(torch.round((self.M / (2 * torch.pi)) * torch.imag(tfr_d[m, n] / tfr[m, n])).item()) # réalocation en fréquence
                    #erification des bornes
                    n_out_of_bounds = (n_hat < 0) or (n_hat >= N)
                    m_out_of_bounds = (m_hat < 0) or (m_hat >= self.M)
                    
                    if n_out_of_bounds or m_out_of_bounds:
                        lost += torch.abs(tfr[m, n]).item() ** 2
                        continue
                    
                    rtfr[m_hat, n_hat] = rtfr[m_hat, n_hat] + torch.abs(tfr[m, n]).item() ** 2 #ajout de l'énergie à la position réalouée
                    
        return rtfr, lost
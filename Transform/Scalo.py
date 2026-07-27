import torch
import math 
from Transform.Transformation import Transformation


class Scalo(Transformation):
    def __init__(self, M, Ts=1.0, w0=5.0, T_param=1.0, as_range=[0.05, 1.5], gamma_K=1e-5, is_freq=0, eps=1e-6):

        super().__init__(M, eps)
        self.M = M
        self.Ts = Ts
        self.w0 = w0
        self.T_param = T_param
        self.as_range = as_range
        self.gamma_K = gamma_K
        self.is_freq = is_freq

        self.as_axis = self._a_axis(self.M, self.as_range, self.is_freq)

    def _a_axis(self, M, as_range, method):
        if method == 0:  
            start, end = math.log10(as_range[0]), math.log10(as_range[1])
            return torch.logspace(start, end, steps=M, base=10.0)
        elif method == 1:  
            return 1.0 / torch.linspace(as_range[0], as_range[1], steps=M)
        elif method == 2: 
            return torch.linspace(as_range[0], as_range[1], steps=M)
        else:
            raise ValueError("Méthode 'is_freq' non reconnue (doit être 0, 1 ou 2).")

    def forward(self, x):

        x = torch.as_tensor(x, dtype=torch.complex64, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        
        as_v = self.as_axis.to(device) 
        sqrt_pi = math.sqrt(math.pi)

        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        for m in range(self.M):
            a = as_v[m]

            K = round(math.sqrt(2 * math.log(1 / self.gamma_K)) * (self.T_param / self.Ts) * a.item())
            
            if K <= 0:
                K = 1

            k = torch.arange(-K, K + 1, device=device, dtype=torch.float32)
            tau = k * self.Ts

            gauss = torch.exp(- (tau ** 2) / (2 * (self.T_param * a) ** 2))
            oscillation = torch.exp(-1j * self.w0 * tau / a)
            wavelet = gauss * oscillation

            norm_fact = self.Ts / (torch.sqrt(torch.abs(a) * self.T_param * sqrt_pi))
            wavelet_normalized = norm_fact * wavelet


            x_padded = torch.nn.functional.pad(
                x.unsqueeze(0).unsqueeze(0), (K, K), mode="constant", value=0.0
            ).squeeze(0).squeeze(0)

            x_frames = x_padded.unfold(0, 2 * K + 1, 1)[:N]

            tfr[m, :] = torch.sum(x_frames * wavelet_normalized, dim=1)

        return tfr

    def rec(self, tfr):

        device = tfr.device
        N = tfr.shape[1]
        as_v = self.as_axis.to(device)

        K_int = 500
        dw = 0.001
        w = torch.arange(self.w0 - K_int, (self.w0 + K_int) + self.eps, dw, device=device)
        
        integrand = (1.0 / w) * torch.exp(-((self.w0 - w) ** 2 * self.T_param ** 2) / 2.0)
        C_psi = math.sqrt(2 * self.T_param * math.sqrt(math.pi)) * torch.sum(integrand) * dw

        ds = torch.zeros_like(as_v)
        if self.M > 1:
            ds[0] = torch.abs(as_v[1] - as_v[0])
            ds[1:] = torch.abs(torch.diff(as_v))
        else:
            ds[0] = 1.0

        as_mat = as_v.view(-1, 1)  
        ds_mat = ds.view(-1, 1)     

        integrand_reconstruction = tfr * (as_mat ** (-1.5)) * ds_mat
        s_hat = (1.0 / C_psi) * torch.sum(integrand_reconstruction, dim=0)

        return torch.real(s_hat)
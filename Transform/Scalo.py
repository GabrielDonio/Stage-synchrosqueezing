import torch
from Transform.Transformation import Transformation
from mmaxis import a_axis

class Scalo(Transformation):
    def __init__(self, M, T, Ts, w0, eps=1e-6, gamma_K=1e-5, as_range=(0.05, 1.5), is_freq=0, hop_length=1):
        super().__init__(M, eps)
        self.M, self.T, self.Ts, self.w0 = M, T, Ts, w0
        self.hop_length = hop_length
        self.scales = a_axis(M, as_range, method=is_freq)
        self.Cpsi = self._compute_Cpsi()
        self.kernels = self._precompute_kernels() 

    def _compute_Cpsi(self):
        span = 10.0 
        omega = torch.linspace(
            max(self.w0 - span, 1e-4),
            self.w0 + span,
            20000
        )
        return torch.trapz(torch.exp(-0.5 * (omega - self.w0) ** 2) / omega, omega).real
    def _precompute_kernels(self):
        a_max = self.scales.max()
        K_max = int(torch.sqrt(2 * torch.log(torch.tensor(1.0/1e-6))) * (self.T / self.Ts) * a_max)
        self.len_kernel = 2 * K_max + 1
        
        kernels = torch.zeros((self.M, self.len_kernel), dtype=torch.complex64)
        tau = torch.arange(-K_max, K_max + 1) * self.Ts
        
        for m in range(self.M):
            a = self.scales[m]
            norm = self.Ts / torch.sqrt(torch.abs(a) * self.T * torch.sqrt(torch.tensor(torch.pi)))
            gauss = torch.exp(-(tau ** 2) / (2 * (self.T * a) ** 2))
            osc = torch.exp(-1j * self.w0 * tau / a)
            kernels[m, :] = norm * gauss * torch.conj(osc) 
        return kernels

    def forward(self, x):
        device = x.device
        pad = self.len_kernel // 2
        x_padded = torch.nn.functional.pad(x.view(1, -1), (pad, pad), mode='reflect').squeeze(0)
        
        frames = x_padded.unfold(0, self.len_kernel, self.hop_length)
        
        W = torch.matmul(self.kernels.to(device), frames.t().to(device))
        
        return W

    def rec(self, W):
        scales = self.scales.to(W.device)

        if self.M > 1:
            log_scales = torch.log(scales)
            d_log_a = (log_scales[-1] - log_scales[0]) / (self.M - 1)
            delta_a = scales * d_log_a
        else:
            delta_a = torch.tensor([1.0], device=W.device, dtype=scales.dtype)

        delta_a = delta_a.view(-1, 1)
        scales = scales.view(-1, 1)

        x_rec = torch.sum(
            W * (delta_a / (torch.abs(scales) ** 1.5)),
            dim=0
        ) / self.Cpsi

        return x_rec.real
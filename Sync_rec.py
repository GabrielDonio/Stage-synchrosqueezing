import torch
from Reconstruction import Reconstruction

class Sync_rec(Reconstruction):
    def __init__(self, M, window=None, eps=1e-6, L=10, isgab=False):
        super().__init__(M, eps)
        self.L = L
        self.isgab = isgab
        if window is None:
            self.len_win = 0
        else:
            self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
            self.len_win = window.shape[0]

    def forward(self, rtfr):
        rtfr = torch.as_tensor(rtfr)
        device = rtfr.device
        M, N = rtfr.shape
        
        center = self.len_win // 2
        h0 = self.window[center].to(device)
        if torch.abs(h0) < self.eps:
            raise ValueError("Le centre de la fenêtre h(0) est nul")
        s_hat = torch.zeros(N, dtype=torch.complex64, device=device)
        if self.isgab:
            coef = torch.sqrt(torch.tensor(2.0 * torch.pi, device=device)) * self.L / self.M
            for n in range(N):
                s_hat[n] = coef * torch.sum(rtfr[:, n])
        else:
            for n in range(N):
                s_hat[n] = torch.sum(rtfr[:, n]) / (self.M * torch.conj(h0))
                
        return s_hat
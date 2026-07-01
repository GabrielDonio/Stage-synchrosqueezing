import torch 
from Reconstruction import Reconstruction
from mmaxis import m_axis
class Stft_rec(Reconstruction):
    """
    implementation of the reconstruction of the STFT.

    Args:
        M (int): Number of frequency bins.
        eps (float): Threshold for the reconstruction.
        window (torch.Tensor): The window function.
    """
    def __init__(self, M,window=None, eps=1e-6, L=10, isgab=True):
        super().__init__(M, eps)
        self.L = L
        self.isgab = isgab
        if window is None:
            self.len_win = 0
        else:
            self.window = torch.as_tensor(window, dtype=torch.float32, device=window.device).reshape(-1)
            self.len_win = window.shape[0]
    def forward(self, tfr):
        tfr = torch.as_tensor(tfr)
        device = tfr.device
        M, N = tfr.shape

        s_hat = torch.zeros(N, dtype=torch.complex64, device=device)
        mm = m_axis(self.M, device=device)

        if self.isgab:
            coef = torch.sqrt(torch.tensor(2.0 * torch.pi, device=device)) * self.L / self.M
            for n in range(N):
                phase = torch.exp(1j * 2 * torch.pi * n * mm / self.M)
                s_hat[n] = coef * torch.sum(tfr[:, n] * phase)
        else:
            center = self.len_win // 2
            h0 = self.window[center].to(device)
            

            if torch.abs(h0) < self.eps:
                raise ValueError("Le centre de la fenêtre h(0) est nul")
                
            for n in range(N):
                phase = torch.exp(1j * 2 * torch.pi * n * mm / self.M)
                s_hat[n] = torch.sum(tfr[:, n] * phase) / (self.M * h0)
        return s_hat

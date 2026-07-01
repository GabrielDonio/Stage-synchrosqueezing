import torch
from mmaxis import m_axis
from Transformation import Transformation

class Tfrgabhop(Transformation):
    """
    implementation of the Gabor transform with hop length.

    Args:
        M (int): Number of frequency bins.
        hop_length (int): Hop length for the Gabor transform.
        eps (float): Threshold for the Gabor transform.
        L (float): Width of the Gaussian window.
        gamma_K (float): Threshold for the Gaussian window.
    Returns:
        tfr (torch.Tensor): Time-frequency representation.
    """
    def __init__(self, M,hop_length, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.L = L
        self.gamma_K = gamma_K #seuillage pour la fenetre de gabor

        self.hop_length = hop_length
        self.K = 2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))
        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.B = -1j * 2 * torch.pi / self.M
        self.C = -1 / (2 * self.L**2)
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device      
        num_frames = len(range(0, N, self.hop_length))
        tfr = torch.zeros((self.M, num_frames), dtype=torch.complex64, device=device)
        
        window = 2 * self.L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / self.gamma_K))) #calcul de la taille de la fenêtre de Gabor

        mm = m_axis(self.M, device=device)
        
        for frame_idx, n in enumerate(range(0, N, self.hop_length)):
            window_min = min(n, int(torch.round(window / 2).item()))
            window_max = min(N - 1 - n, int(torch.round(window / 2).item()))
            k = torch.arange(-window_min , window_max + 1, device=device)
            k2 = k ** 2

            g = self.A * torch.exp(self.C * k2)

            for m in range(self.M):
                exp_B_mm_k = torch.exp(self.B * mm[m] * k)
                exp_B_mm_nn = torch.exp(self.B * mm[m] * n)

                idx = n + k
                valid = (idx >= 0) & (idx < N)

                x_slice = x[idx[valid]]
                g_valid = g[valid]
                exp_B_mm_k_valid = exp_B_mm_k[valid]

                tfr[m, frame_idx] = exp_B_mm_nn * torch.sum(x_slice * g_valid * exp_B_mm_k_valid)
                
        return tfr

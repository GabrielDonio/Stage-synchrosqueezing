import torch
from Transformation.Transformation import Transformation

class Tfrgabhop(Transformation):
    """
    Implementation of the Gabor transform with hop length.
    """
    def __init__(self, M, hop_length, eps=1e-6, L=10, gamma_K=1e-4):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K
        self.hop_length = hop_length

        self.K_val = int(torch.round(2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))).item())
        self.half_K = self.K_val // 2
        self.len_win = 2 * self.half_K + 1

        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.C = -1 / (2 * self.L**2)

        k = torch.arange(-self.half_K, self.half_K + 1, dtype=torch.float32)
        self.g = self.A * torch.exp(self.C * (k ** 2))
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device      

        g_v = self.g.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.half_K, self.half_K), mode='reflect')
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, N, self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]

        tfr_segments = x_frames * g_v

        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t() 
        
        m_vec = torch.arange(self.M, device=device).view(-1, 1)
        n_vec_row = n_vec.view(1, -1) 
        
        phase_correction = torch.exp(1j * 2 * torch.pi / self.M * m_vec * (self.half_K - n_vec_row))
        
        tfr = tfr * phase_correction
                
        return tfr
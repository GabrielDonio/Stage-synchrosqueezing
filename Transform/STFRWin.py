import torch
from mmaxis import m_axis
from Transform.Transformation import Transformation


class Stfrwin(Transformation):
    def __init__(self, M, window, hop_length=1, eps=0.01):
        super().__init__(M, eps)
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]
        self.hop_length = hop_length
        self.dw = -torch.gradient(self.window)[0]#derive de la fenetre
        
        self.center = self.len_win // 2

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.complex64, device=x.device).reshape(-1)
        self.N_input = x.shape[0] 
        device = x.device

        w_v = self.window.to(device)
        dw_v = self.dw.to(device)

        x_padded = torch.nn.functional.pad(x, (self.center, self.center), mode="constant", value=0)

        n_vec = torch.arange(0, self.N_input, self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]

        tfr_segments = x_frames * w_v
        tfr_d_segments = x_frames * dw_v

        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t()
        tfr_d = torch.fft.fft(tfr_d_segments, n=self.M, dim=1).t()

        #m_vec = torch.arange(self.M, device=device).view(-1, 1)
        m_vec = m_axis(self.M, device=device).view(-1, 1)
        fft_phase_correction = torch.exp(2j * torch.pi * m_vec * self.center / self.M)

        tfr = tfr * fft_phase_correction
        tfr_d = tfr_d * fft_phase_correction

        magnitude = torch.abs(tfr)
        max_per_col, _ = torch.max(magnitude, dim=0, keepdim=True)
        mask = magnitude > (self.eps * max_per_col)

        v_m = torch.zeros_like(tfr, dtype=torch.float32)
        v_m[mask] = torch.imag(tfr_d[mask] / tfr[mask])#application de la formule de synchrosqueezing 
        m_indices = torch.arange(self.M, device=device).view(-1, 1).expand(self.M, num_frames)
        m_hat = m_indices + torch.round(v_m * self.M / (2 * torch.pi)).long() #on calcul les points de synchrosqueezing

        valid_bounds = (m_hat >= 0) & (m_hat < self.M) & mask
        lost_mask = mask & (~valid_bounds)
        lost = torch.sum(magnitude[lost_mask] ** 2).item()#energie perdue lors du synchrosqueezing numérique

        rtfr = torch.zeros((self.M, num_frames), dtype=torch.complex64, device=device)

        indices_valides = torch.nonzero(valid_bounds, as_tuple=True)
        m_dest = m_hat[indices_valides]
        n_dest = indices_valides[1]
        flat_dest_indices = m_dest * num_frames + n_dest
        
        flat_valeurs = tfr[indices_valides]

        rtfr.view(-1).index_add_(0, flat_dest_indices, flat_valeurs)

        return rtfr, lost

    def rec(self, rtfr):

        device = rtfr.device
        M, num_frames = rtfr.shape

        if self.hop_length != 1:
            print("Pas de rec à hop>1")

        h_0 = self.window[self.center].to(device)

        scale = 1.0 / (M * h_0)

        x_reconstructed = torch.sum(rtfr, dim=0).real * scale
        
        if hasattr(self, 'N_input') and self.hop_length == 1:
            return x_reconstructed[:self.N_input]

        return x_reconstructed
import torch
from mmaxis import m_axis
from Transform.Transformation import Transformation


class Tfrgab(Transformation):
    """
    Gabor time-frequency representation.
    """
    def __init__(self, M, eps=1e-6, L=10, gamma_K=1e-4, hop_length=1):
        super().__init__(M, eps)

        self.M = M
        self.L = L
        self.gamma_K = gamma_K
        self.hop_length = hop_length
        
        # On tronque la fenetre de gabor à un seuil choisie.
        self.K_val = int(
            torch.round(
                2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))
            ).item()
        )
        self.half_K = self.K_val // 2
        self.len_win = 2 * self.half_K + 1

        self.A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * self.L)
        self.C = -1 / (2 * self.L**2)

        k = torch.arange(-self.half_K, self.half_K + 1, dtype=torch.float32)
        self.g = self.A * torch.exp(self.C * (k ** 2))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.complex64, device=x.device).reshape(-1)
        self.N_input = x.shape[0]
        device = x.device

        g_v = self.g.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.half_K, self.half_K), mode="reflect")# On va utiliser reflect pour garder les fenetres centrées même aux bords du signal.
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, x.shape[0], self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        
        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]#on deccoupe les frames du signal

        tfr_segments = x_frames * g_v 

        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t()  #Application de la FFt sur caque frame
        """
        m_vec = m_axis(self.M, device=device).view(-1, 1)
        n_vec_row = n_vec.view(1, -1)

        phase_correction = torch.exp(
            1j * 2 * torch.pi / self.M * m_vec * (self.half_K - n_vec_row)
        )"""
        #tfr = tfr #* phase_correction
            
        return tfr
    def rec(self, tfr):
        device = tfr.device
        M, num_frames = tfr.shape
        g_v = self.g.to(device)

        if not hasattr(self, "N_input"):
            raise ValueError("N_input is not set. Please call forward() before rec().")

        """n_vec = torch.arange(0, num_frames * self.hop_length, self.hop_length, device=device)[:num_frames]
        m_vec = m_axis(M, device=device).view(-1, 1)

        #phase_correction = torch.exp(
        #    1j * 2 * torch.pi / M * m_vec * (self.half_K - n_vec.view(1, -1))
        #)"""
        tfr_origin = tfr #* torch.conj(phase_correction)

        tfr_segments = torch.fft.ifft(tfr_origin.t(), n=M, dim=1) #application de la IFFT sur chaque frame
        tfr_segments = tfr_segments[:, :self.len_win]

        tfr_segments = tfr_segments #* g_v

        len_padded_output = (num_frames - 1) * self.hop_length + self.len_win#longueur du signal reconstruit après overlap-add
        
        idx = torch.arange(self.len_win, device=device).unsqueeze(0) + (
            torch.arange(num_frames, device=device).unsqueeze(1) * self.hop_length
        )
        idx_flat = idx.view(-1)

        x_reconstructed = torch.zeros(len_padded_output, device=device, dtype=torch.complex64)
        x_reconstructed.index_add_(0, idx_flat, tfr_segments.reshape(-1))#on reconstruit le signal en sommant les frames avec overlap-add

        window_sum = torch.zeros(len_padded_output, device=device, dtype=torch.float32)
        g_squared = (g_v).repeat(num_frames)#pour la normalisation de la fenetre
        window_sum.index_add_(0, idx_flat, g_squared)

        mask = window_sum > self.eps
        x_reconstructed[mask] /= window_sum[mask].to(x_reconstructed.dtype)#la normalisation

        x_final = x_reconstructed[self.half_K : self.half_K + self.N_input]

        return x_final.real
    def rec_mor(self, tfr):
        if self.hop_length == 1:
        
            device = tfr.device
            M, num_frames = tfr.shape
            g_v = self.g.to(device)

            if not hasattr(self, "N_input"):
                raise ValueError("N_input is not set. Please call forward() before rec().")

            n_vec = torch.arange(0, num_frames * self.hop_length, self.hop_length, device=device)[:num_frames]

            m_vec = m_axis(M, device=device).view(-1, 1)

            #phase_correction = torch.exp(
            #    1j * 2 * torch.pi / M * m_vec * (self.half_K - n_vec.view(1, -1))
            #)
                
            tfr_origin = tfr #* torch.conj(phase_correction)

            tfr_segments = torch.fft.ifft(tfr_origin.t(), n=M, dim=1)
            tfr_segments = tfr_segments[:, :self.len_win]

            x_at_t = tfr_segments[:, self.half_K]
            h_0 = g_v[self.half_K]
                    
            if torch.abs(h_0) < self.eps:
                raise ValueError("Le centre de la fenêtre de Gabor est nul ou trop petit.")
                        
            x_reconstructed = x_at_t / h_0

            return x_reconstructed[:self.N_input].real
        else:
            raise ValueError("rec_mor is only applicable when hop_length is 1.")
    

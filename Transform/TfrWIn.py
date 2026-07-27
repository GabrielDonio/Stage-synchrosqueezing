import torch
from mmaxis import m_axis
from Transform.Transformation import Transformation


class TfrWin(Transformation):
    def __init__(self, M, window, hop_length=1, eps=1e-6):
        super().__init__(M, eps)
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32).reshape(-1)
        self.len_win = self.window.shape[0]
        self.hop_length = hop_length

        self.center = self.len_win // 2
        self.left = self.center
        self.right = self.len_win - self.center - 1

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.complex64, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device

        w_v = self.window.to(device)

        x_3d = x.unsqueeze(0).unsqueeze(0)
        x_padded_3d = torch.nn.functional.pad(x_3d, (self.left, self.right), mode="reflect")
        x_padded = x_padded_3d.squeeze(0).squeeze(0)

        n_vec = torch.arange(0, N, self.hop_length, device=device)
        num_frames = n_vec.shape[0]

        x_frames = x_padded.unfold(0, self.len_win, self.hop_length)[:num_frames]
        tfr_segments = x_frames * w_v

        tfr = torch.fft.fft(tfr_segments, n=self.M, dim=1).t()
        

        #m_vec = torch.arange(self.M, device=device).view(-1, 1)
        #n_vec_row = n_vec.view(1, -1)

        #phase_correction = torch.exp(
        #    1j * 2 * torch.pi / self.M * m_vec * (self.left - n_vec_row)
        #)

        return tfr #* phase_correction
    def rec(self, tfr):  # overlap-add reconstruction 
        device = tfr.device
        M, num_frames = tfr.shape
        w_v = self.window.to(device)

        N_target = getattr(self, 'N_input', (num_frames - 1) * self.hop_length + 1)

        #n_vec = torch.arange(0, num_frames * self.hop_length, self.hop_length, device=device)[:num_frames]
        #m_vec = torch.arange(M, device=device).view(-1, 1) 
        
        #phase_correction = torch.exp(
        #    1j * 2 * torch.pi / M * m_vec * (self.left - n_vec.view(1, -1))
        #)
        tfr_origin = tfr #* torch.conj(phase_correction)

        tfr_segments = torch.fft.ifft(tfr_origin.t(), n=M, dim=1)
        tfr_segments = tfr_segments[:, :self.len_win] 

        tfr_segments = tfr_segments  

        len_padded_output = (num_frames - 1) * self.hop_length + self.len_win
        
        idx = torch.arange(self.len_win, device=device).unsqueeze(0) + (
            torch.arange(num_frames, device=device).unsqueeze(1) * self.hop_length
        )
        idx_flat = idx.view(-1)

        x_reconstructed = torch.zeros(len_padded_output, device=device, dtype=torch.complex64)
        x_reconstructed.index_add_(0, idx_flat, tfr_segments.reshape(-1))

        window_sum = torch.zeros(len_padded_output, device=device, dtype=torch.float32)
        w_squared = (w_v).repeat(num_frames) 
        window_sum.index_add_(0, idx_flat, w_squared)

        mask = window_sum > self.eps
        x_reconstructed[mask] /= window_sum[mask].to(x_reconstructed.dtype)

        x_final = x_reconstructed[self.left : self.left + N_target]

        return x_final
    def rec_mor(self, tfr):
        if self.hop_length == 1:
            device = tfr.device
            M, num_frames = tfr.shape
            w_v = self.window.to(device)

            N_target = getattr(self, 'N_input', (num_frames - 1) * self.hop_length + 1)

            #n_vec = torch.arange(0, num_frames * self.hop_length, self.hop_length, device=device)[:num_frames]
            #m_vec = torch.arange(M, device=device).view(-1, 1) 
            
            #phase_correction = torch.exp(
            #    1j * 2 * torch.pi / M * m_vec * (self.left - n_vec.view(1, -1))
            #)
            tfr_origin = tfr #* torch.conj(phase_correction)

            tfr_segments = torch.fft.ifft(tfr_origin.t(), n=M, dim=1)
            tfr_segments = tfr_segments[:, :self.len_win] 
            
            x_at_t = tfr_segments[:, self.left]
            h_0 = w_v[self.left]

            if torch.abs(h_0) < self.eps:
                raise ValueError("Le centre de la fenêtre est nul ou trop proche de zéro.")
                    
            x_reconstructed = x_at_t / h_0
            return x_reconstructed[:N_target]

        else:
            print("rec_mor is only applicable when hop_length is 1.")
    

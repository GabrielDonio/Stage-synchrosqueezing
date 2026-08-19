import torch
import math
from Transform.Transformation import Transformation

class tfrhsgab2(Transformation):
    def __init__(self, M=None, L=10, gamma_K=1e-4, q_method=2):
        super().__init__(M)
        self.M = M
        self.L = L
        self.gamma_K = gamma_K
        self.q_method = q_method

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.flatten().to(dtype=torch.complex128)
        device = x.device
        
        N = x.shape[0]
        M = self.M if self.M is not None else N
        
        eps = 1e-12 
        
        lost = 0.0
        lost2 = torch.zeros(M, dtype=torch.complex128, device=device)
        tfr = torch.zeros((M, N), dtype=torch.complex128, device=device)
        stfr = torch.zeros((M, N), dtype=torch.complex128, device=device)
        
        K = 2 * self.L * math.sqrt(2 * math.log(1 / self.gamma_K))
        A = 1 / (math.sqrt(2 * math.pi) * self.L)
        B = -1j * 2 * math.pi / M
        C = -1 / (2 * self.L**2)
        
        mm = torch.arange(M, device=device)
        
        for n in range(N):
            k_min = min(n, round(K/2)) 
            k_max = min(N - 1 - n, round(K/2))
            
            k = torch.arange(-k_min, k_max + 1, device=device)
            k2 = k**2
            
            g_Ts = A * torch.exp(C * k2)
            tg = -k * g_Ts
            dg_Ts2 = -1 / (self.L**2) * tg
            
            t2g_Tsm1 = k2 * g_Ts
            tdg_Ts = -k2 / (self.L**2) * g_Ts
            d2g_Ts3 = (-1 / (self.L**2) + k2 / (self.L**4)) * g_Ts
            
            x_slice = x[n + k]

            phase_shift = torch.exp(B * mm * (n - k_min))
            
            # FFTs
            tfr_n = torch.fft.fft(x_slice * g_Ts, n=M) * phase_shift
            tfr_t_n = torch.fft.fft(x_slice * tg, n=M) * phase_shift
            tfr_d_n = torch.fft.fft(x_slice * dg_Ts2, n=M) * phase_shift
            tfr_td_n = torch.fft.fft(x_slice * tdg_Ts, n=M) * phase_shift
            tfr_t2_n = torch.fft.fft(x_slice * t2g_Tsm1, n=M) * phase_shift
            tfr_d2_n = torch.fft.fft(x_slice * d2g_Ts3, n=M) * phase_shift
            
            tfr[:, n] = tfr_n
            
            mask = torch.abs(tfr_n) > eps
            
            if not mask.any():
                continue
                
            m_valid = mm[mask]
            
            tfr_v = tfr_n[mask]
            tfr_t_v = tfr_t_n[mask]
            tfr_d_v = tfr_d_n[mask]
            tfr_td_v = tfr_td_n[mask]
            tfr_t2_v = tfr_t2_n[mask]
            tfr_d2_v = tfr_d2_n[mask]
            
            n_tilde = n - (tfr_t_v / tfr_v)
            n_hat = n - torch.round(torch.real(tfr_t_v / tfr_v))
            m_hat = m_valid + torch.round((M / (2 * math.pi)) * torch.imag(tfr_d_v / tfr_v))
            
            dt_hat_dt = tfr_td_v - tfr_t_v * tfr_d_v
            
            q_hat = torch.zeros_like(dt_hat_dt)
            q_mask = torch.abs(dt_hat_dt) > eps
            
            if q_mask.any():
                v_tfr = tfr_v[q_mask]
                v_tfr_t = tfr_t_v[q_mask]
                v_tfr_d = tfr_d_v[q_mask]
                v_tfr_td = tfr_td_v[q_mask]
                v_tfr_d2 = tfr_d2_v[q_mask]
                v_tfr_t2 = tfr_t2_v[q_mask]
                
                if self.q_method == 2:
                    num = (v_tfr_d2 * v_tfr) - (v_tfr_d**2)
                    den = (v_tfr_t * v_tfr_d) - (v_tfr_td * v_tfr)
                else: 
                    num = (v_tfr_td * v_tfr) + (v_tfr**2) - (v_tfr_t * v_tfr_d)
                    den = (v_tfr_t**2) - (v_tfr_t2 * v_tfr)
                    
                q_hat[q_mask] = num / den

            n_hat_q = n_hat.clone().to(torch.float64)
            
            valid_q = (torch.abs(q_hat) > eps) & (torch.abs(1 / (q_hat + 1e-16)) > eps)
            
            if valid_q.any():
                q_v = q_hat[valid_q]
                imag_q = torch.imag(q_v) + 1e-16 
                
                term1 = torch.imag(q_v * n_tilde[valid_q]) / imag_q
                term2 = torch.round((2 * math.pi / M) * (1 / imag_q) * (m_valid[valid_q] - m_hat[valid_q]))
                
                n_hat_q_val = torch.round(term1 + term2)

                nan_mask = torch.isnan(n_hat_q_val)
                n_hat_q_val[nan_mask] = n_hat[valid_q][nan_mask].to(torch.float64)
                
                n_hat_q[valid_q] = n_hat_q_val
                
            n_hat_q = n_hat_q.to(torch.long)
            
            in_bounds = (n_hat_q >= 0) & (n_hat_q < N)
            
            if (~in_bounds).any():
                out_idx = (~in_bounds).nonzero().squeeze(-1)
                lost += torch.sum(torch.abs(tfr_v[out_idx])**2).item()
                lost2.scatter_add_(0, m_valid[out_idx], tfr_v[out_idx])
            
            if in_bounds.any():
                in_m = m_valid[in_bounds]
                in_n = n_hat_q[in_bounds]
                in_tfr = tfr_v[in_bounds]
                
                stfr.index_put_((in_m, in_n), in_tfr, accumulate=True)
                
        return tfr, stfr, lost, lost2

    def rec(self, stfr):
        if not isinstance(stfr, torch.Tensor):
            stfr = torch.tensor(stfr)
            
        N = stfr.shape[1]
        M = self.M if self.M is not None else stfr.shape[0]
            
        if N <= M:
            s_hat = torch.fft.ifft(stfr.sum(dim=1))
            return s_hat[:N].flatten()
        
        hop = M // 2
        s_hat = torch.zeros(N, dtype=torch.complex128, device=stfr.device)
        norm = torch.zeros(N, dtype=torch.float64, device=stfr.device)
        window = torch.hann_window(M, device=stfr.device, dtype=torch.float64)
        
        for i in range(0, N, hop):
            end_idx = min(i + M, N)
            chunk_size = end_idx - i

            stfr_chunk = stfr[:, i:end_idx]
            
            S_chunk = stfr_chunk.sum(dim=1)

            A = torch.fft.ifft(S_chunk)
            
            n_indices = torch.arange(i, end_idx, device=stfr.device)
            local_signal = A[n_indices % M]
            
            win_chunk = window[:chunk_size]
            s_hat[i:end_idx] += local_signal * win_chunk
            norm[i:end_idx] += win_chunk
            
        mask = norm > 1e-10
        s_hat[mask] /= norm[mask]
        
        return s_hat.flatten()
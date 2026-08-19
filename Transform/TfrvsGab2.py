import torch
import math
from Transform.Transformation import Transformation

class tfrvsgab2(Transformation):
    def __init__(self, M=None, L=10.0, q_method=2, if_method=1, gamma_K=1e-4, q_threshold=1e-4):
        super().__init__(M)
        self.L = L
        self.q_method = q_method
        self.if_method = if_method
        self.gamma_K = gamma_K
        self.q_threshold = q_threshold

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.flatten().to(dtype=torch.complex128)
        device = x.device
        
        N = x.shape[0]
        M = self.M if self.M is not None else N
        eps = 1e-12
        
        lost = 0.0
        tfr = torch.zeros((M, N), dtype=torch.complex128, device=device)
        stfr = torch.zeros((M, N), dtype=torch.complex128, device=device)
        q_hatmap = torch.zeros((M, N), dtype=torch.complex128, device=device)
        if_hatmap = torch.zeros((M, N), dtype=torch.float64, device=device)
        
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
            
            tfr_n = torch.fft.fft(x_slice * g_Ts, n=M) * phase_shift
            tfr_t_n = torch.fft.fft(x_slice * tg, n=M) * phase_shift
            tfr_d_n = torch.fft.fft(x_slice * dg_Ts2, n=M) * phase_shift
            tfr_td_n = torch.fft.fft(x_slice * tdg_Ts, n=M) * phase_shift
            tfr_t2_n = torch.fft.fft(x_slice * t2g_Tsm1, n=M) * phase_shift
            tfr_d2_n = torch.fft.fft(x_slice * d2g_Ts3, n=M) * phase_shift
            
            if self.q_method == 4:
                t2dg = k2 * dg_Ts2
                t3g_Tsm2 = -k * t2g_Tsm1
                tfr_t2d_n = torch.fft.fft(x_slice * t2dg, n=M) * phase_shift
                tfr_t3_n = torch.fft.fft(x_slice * t3g_Tsm2, n=M) * phase_shift
            
            tfr[:, n] = tfr_n
            
            mask = torch.abs(tfr_n) > eps
            if not mask.any():
                continue
                
            m_valid = mm[mask]
            v_tfr = tfr_n[mask]
            v_tfr_t = tfr_t_n[mask]
            v_tfr_d = tfr_d_n[mask]
            v_tfr_td = tfr_td_n[mask]
            v_tfr_t2 = tfr_t2_n[mask]
            v_tfr_d2 = tfr_d2_n[mask]
            
            tfr_t_div = v_tfr_t / v_tfr
            tfr_d_div = v_tfr_d / v_tfr
            
            round_tfr_t_div = torch.round(tfr_t_div.real) + 1j * torch.round(tfr_t_div.imag)
            
            n_tilde = n - round_tfr_t_div
            n_tilde2 = n - tfr_t_div
        
            n_hat = n_tilde.real
            m_hat = m_valid + torch.round((M / (2 * math.pi)) * tfr_d_div.imag)
            
            n_hat2 = n_tilde2.real
            m_hat2 = m_valid + (M / (2 * math.pi)) * tfr_d_div.imag
            
            q_hat = torch.zeros_like(v_tfr)
            alpha_hat = torch.zeros_like(v_tfr.real)
            
            if self.q_method == 1:
                alpha_denum = torch.imag(v_tfr_t * torch.conj(v_tfr))
                q_mask = torch.abs(alpha_denum) > self.q_threshold
                alpha_hat[q_mask] = torch.real(v_tfr_d[q_mask] * torch.conj(v_tfr[q_mask])) / alpha_denum[q_mask]
                q_hat = 1j * alpha_hat
                
            elif self.q_method == 2:
                q_denum = v_tfr_t * v_tfr_d - v_tfr_td * v_tfr
                q_mask = torch.abs(q_denum) > self.q_threshold
                q_hat[q_mask] = (v_tfr_d2[q_mask] * v_tfr[q_mask] - v_tfr_d[q_mask]**2) / q_denum[q_mask]
                alpha_hat = torch.imag(q_hat)
            elif self.q_method == 3:
                q_denum = v_tfr_t**2 - v_tfr_t2 * v_tfr
                q_num = (v_tfr_td * v_tfr - v_tfr_t * v_tfr_d + v_tfr**2)
                q_mask = torch.abs(q_denum) > (self.q_threshold * torch.abs(v_tfr)**2)
                
                q_hat[q_mask] = q_num[q_mask] / q_denum[q_mask]
                
                aberrant_mask = torch.abs(q_hat) > (1.0 / self.q_threshold)
                q_hat[aberrant_mask] = 0.0
                
                alpha_hat = torch.imag(q_hat)
                            
            elif self.q_method == 4:
                v_tfr_t2d = tfr_t2d_n[mask]
                v_tfr_t3 = tfr_t3_n[mask]
                
                Ar = torch.zeros((len(v_tfr), 3, 3), dtype=torch.complex128, device=device)
                Ar[:, 0, 0], Ar[:, 0, 1], Ar[:, 0, 2] = v_tfr_t2, -v_tfr_t, v_tfr
                Ar[:, 1, 0], Ar[:, 1, 1], Ar[:, 1, 2] = 2 * v_tfr_t + v_tfr_t2d, -v_tfr_td, v_tfr_d
                Ar[:, 2, 0], Ar[:, 2, 1], Ar[:, 2, 2] = v_tfr_t3, -v_tfr_t2, v_tfr_t
                
                ur = torch.zeros((len(v_tfr), 3, 1), dtype=torch.complex128, device=device)
                ur[:, 0, 0] = v_tfr_d
                ur[:, 1, 0] = v_tfr_d2
                ur[:, 2, 0] = v_tfr_td + v_tfr
                
                det_Ar = torch.linalg.det(Ar)
                q_mask = torch.abs(det_Ar) > self.q_threshold
                
                if q_mask.any():
                    Ar_valid = Ar[q_mask]
                    ur_valid = ur[q_mask]
                    
                    Ar_inv = torch.linalg.pinv(Ar_valid)
                    x_sol = torch.matmul(Ar_inv, ur_valid)
                    
                    rx = x_sol[:, 0, 0]
                    q_val = x_sol[:, 1, 0] - 2 * rx * (n - n_tilde[q_mask])
                    
                    limit_mask = torch.abs(q_val) <= 1 / self.q_threshold
                    valid_idx = q_mask.nonzero(as_tuple=True)[0][limit_mask]
                    
                    q_hat[valid_idx] = q_val[limit_mask]
                    
                alpha_hat = torch.imag(q_hat)

            if self.if_method == 1:
                m_hat_q = m_hat + torch.round((M / (2 * math.pi)) * alpha_hat * (n - n_hat))
                if_val = (m_hat2 + (M / (2 * math.pi)) * alpha_hat * (n - n_hat2)) / M
            elif self.if_method == 2:
                term_q = (M / (2 * math.pi)) * q_hat * (n - n_tilde)
                m_hat_q = m_hat + torch.round(term_q.imag)
                
                term_q2 = (M / (2 * math.pi)) * q_hat * (n - n_tilde2)
                if_val = (m_hat2 + term_q2.imag) / M
            else:
                m_hat_q = m_hat
                if_val = m_hat2 / M
                
            m_hat_q = m_hat_q.to(torch.long)
            
            q_hatmap[m_valid, n] = q_hat
            if_hatmap[m_valid, n] = if_val
            
            in_bounds = (m_hat_q >= 0) & (m_hat_q < M)
            
            if (~in_bounds).any():
                out_idx = (~in_bounds).nonzero().squeeze(-1)
                lost += torch.sum(torch.abs(v_tfr[out_idx])**2).item()
                
            if in_bounds.any():
                in_m_q = m_hat_q[in_bounds]
                in_v_tfr = v_tfr[in_bounds]
                in_m_valid = m_valid[in_bounds]
                
                phase_stfr = torch.exp(1j * 2 * math.pi * in_m_valid * n / M)
                val_to_add = (in_v_tfr / (2 * math.pi)) * phase_stfr
                
                n_indices = torch.full_like(in_m_q, n)
                stfr.index_put_((in_m_q, n_indices), val_to_add, accumulate=True)
                
        return tfr, stfr, lost, q_hatmap, if_hatmap

    def rec(self, stfr):
        if not isinstance(stfr, torch.Tensor):
            stfr = torch.tensor(stfr)
            
        M = self.M if self.M is not None else stfr.shape[0]
        s_hat = stfr.sum(dim=0) * ((2 * math.pi)**1.5) * self.L / M
        
        return s_hat.flatten()
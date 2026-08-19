import torch
from Transform.STFRGab import Stfrgab
from Transform.STFRWin import Stfrwin
import matplotlib.pyplot as plt
def generate_signal_1(Fs=1000.0, complex_signal=False, device='cpu'):
    Nchirp = 440
    loc_impulse1 = 15  
    loc_impulse2 = 40
    val_impulse = 10.0
    
    t1 = torch.arange(1, Nchirp + 1, dtype=torch.float32, device=device) / Fs
    w1 = 2 * torch.pi * 3
    a1 = 50.0
    p1 = torch.pi
    freq_sin = 355.0
    
    f_inst = freq_sin + a1 * torch.cos(w1 * t1 + p1)
    phi_t = torch.cumsum(f_inst, dim=0) / Fs
    s0 = torch.cos(2 * torch.pi * phi_t)
    
    t_chirp = torch.arange(Nchirp, dtype=torch.float32, device=device)
    fmconst = torch.exp(1j * 2 * torch.pi * 0.1 * t_chirp)
    f1, f2 = 0.12, 0.3
    phase_fmlin = 2 * torch.pi * (f1 * t_chirp + (f2 - f1) / (2 * Nchirp) * (t_chirp ** 2))
    fmlin = torch.exp(1j * phase_fmlin)
    
    s_core = fmconst + fmlin + s0
    
    N = 500
    pad_len = N - Nchirp
    s = torch.cat([torch.zeros(pad_len, dtype=torch.complex64, device=device), s_core])
    
    s[loc_impulse1 - 1] = val_impulse
    s[loc_impulse2 - 1] = val_impulse
    
    if not complex_signal:
        s = torch.real(s)
        
    return s
def gaussian_window(L, gamma_K):
    K_val = int(
        torch.round(
            2 * L * torch.sqrt(torch.tensor(2.0) * torch.log(torch.tensor(1.0 / gamma_K)))
        ).item()
    )
    half_K = K_val // 2
    k = torch.arange(-half_K, half_K + 1, dtype=torch.float32)
    A = 1 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * L)
    C = -1 / (2 * L**2)
    return A * torch.exp(C * (k ** 2))

device = "cuda" if torch.cuda.is_available() else "cpu"

x = generate_signal_1(Fs=1000.0, complex_signal=True, device=device)

M = 1024
L = 10
gamma_K = 1e-4
hop_length = 1
eps = 1e-6

window = gaussian_window(L, gamma_K)

gab = Stfrgab(M=M, hop_length=hop_length, eps=eps, L=L, gamma_K=gamma_K).to(device)
win = Stfrwin(M=M, window=window, hop_length=hop_length, eps=eps).to(device)

rtfr_gab, lost_gab = gab.forward(x)
rtfr_win, lost_win = win.forward(x)

border = gab.half_K
rtfr_gab_c = rtfr_gab[:, border:-border]
rtfr_win_c = rtfr_win[:, border:-border]

diff = rtfr_win_c - rtfr_gab_c
rel_diff = torch.linalg.norm(diff) / (torch.linalg.norm(rtfr_gab_c) + 1e-12)
mean_abs_diff = torch.mean(torch.abs(diff))
max_abs_diff = torch.max(torch.abs(diff))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(torch.abs(rtfr_gab_c).cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
plt.title('STFR Gabor')
plt.xlabel('Time')
plt.ylabel('Frequency Bin')
plt.subplot(1, 2, 2)
plt.imshow(torch.abs(rtfr_win_c).cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
plt.title('STFR Windowed')  
plt.xlabel('Time')
plt.ylabel('Frequency Bin')
plt.tight_layout()  
plt.show()
x_hat_gab = gab.rec(rtfr_gab)
x_hat_win = win.rec(rtfr_win)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x.cpu().numpy(), label='Original')  
plt.plot(x_hat_gab.cpu().numpy(), label='Gabor')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x.cpu().numpy(), label='Original')  
plt.plot(x_hat_win.cpu().numpy(), label='Windowed')
plt.xlabel('Time')  
plt.ylabel('Amplitude')
plt.legend()    
plt.tight_layout()  
plt.show()  
print("lost gab:", lost_gab)
print("lost win:", lost_win)
print("relative tf diff:", rel_diff.item())
print("mean abs tf diff:", mean_abs_diff.item())
print("max abs tf diff:", max_abs_diff.item())

half_gab = rtfr_gab_c.shape[0] // 2
half_win = rtfr_win_c.shape[0] // 2

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(
    torch.abs(rtfr_gab_c[half_gab:, :]).cpu().numpy(),
    aspect='auto',
    cmap='magma'
)
plt.title('STFR Gabor')
plt.xlabel('Time')
plt.ylabel('Frequency Bin')

plt.subplot(1, 2, 2)
plt.imshow(
    torch.abs(rtfr_win_c[half_win:, :]).cpu().numpy(),
    aspect='auto',
    cmap='magma'
)
plt.title('STFR Windowed')
plt.xlabel('Time')
plt.ylabel('Frequency Bin')

plt.tight_layout()
plt.show()

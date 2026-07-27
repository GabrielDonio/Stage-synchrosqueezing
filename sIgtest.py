import matplotlib.pyplot as plt
import numpy as np
import torch
from Transform.TfrvsGab2 import Tfrvsgab2


Fs = 1000.0  
N = 1000 
t = torch.linspace(0, 1, N)

f0, f1 = 50.0, 350.0  
alpha_real_hz = f1 - f0 


phase = 2 * torch.pi * (f0 * t + 0.5 * alpha_real_hz * t**2)
clean_signal = torch.exp(1j * phase)
P_signal = 1.0 

target_t_idx = N // 2
target_time = t[target_t_idx].item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")

M = 512
transform = Tfrvsgab2(
    M=M, L=15.0, q_method=2, if_method=1, eps=1e-3, q_threshold=1e-3
).to(device)

clean_signal = clean_signal.to(device)


snr_db_range = np.arange(-10, 60, 5) 
num_trials = 20 
mse_alpha_list = []



for snr_db in snr_db_range:
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(P_signal / snr_linear)

    squared_errors = []

    for trial in range(num_trials):
        noise = (
            torch.randn(N, device=device) + 1j * torch.randn(N, device=device)
        ) * (noise_std / np.sqrt(2.0))
        x_noisy = clean_signal + noise

        with torch.no_grad():
            tfr, _, _, q_map, _ = transform(x_noisy)

        tfr_col = torch.abs(tfr[:, target_t_idx])

        ridge_m_idx = torch.argmax(tfr_col).item()

        alpha_est_raw = q_map.imag[ridge_m_idx, target_t_idx].item()
        alpha_est_hz = alpha_est_raw * (Fs**2) / (2 * np.pi)

        sq_err = (alpha_est_hz - alpha_real_hz) ** 2
        squared_errors.append(sq_err)

    mse_val = np.mean(squared_errors)
    mse_alpha_list.append(mse_val)

    mse_db_val = 10 * np.log10(mse_val)
    print(
        f"SNR: {snr_db:5.1f} dB | MSE Alpha: {mse_val:10.4f} (Hz/s)² | MSE: {mse_db_val:6.2f} dB"
    )

mse_alpha_arr = np.array(mse_alpha_list)
mse_alpha_db = 10 * np.log10(mse_alpha_arr)

plt.figure(figsize=(9, 5))

plt.plot(
    snr_db_range,
    mse_alpha_db,
    "o-",
    color="navy",
    linewidth=2,
    markersize=6,
)

plt.title(
    f"performance estimateur",
    fontsize=12,
)
plt.xlabel("SNR (dB)", fontsize=11)
plt.ylabel(r"MSE $[10 \log_{10}((\text{Hz/s})^2)]$ (dB)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()
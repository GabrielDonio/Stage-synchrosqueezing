import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from Transform.tfrvsGab2 import tfrvsgab2


def generate_signal_2(Fs=1000.0, complex_signal=True, device='cpu'):
    Nchirp = 220
    t_chirp = torch.arange(Nchirp, dtype=torch.float32, device=device)
    f1, f2 = 0.12, 0.3
    phase_fmlin = 2 * torch.pi * (f1 * t_chirp + (f2 - f1) / (2 * Nchirp) * (t_chirp ** 2))
    fmlin = torch.exp(1j * phase_fmlin)

    N = 500
    pad_len = N - Nchirp
    s = torch.cat([torch.zeros(pad_len, dtype=torch.complex64, device=device), fmlin])
    if not complex_signal:
        s = torch.real(s)

    return s, phase_fmlin


def add_awgn(signal, snr_db):
    signal_power = torch.mean(torch.abs(signal) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise_std = torch.sqrt(noise_power / 2.0)
    noise_real = torch.randn_like(signal.real) * noise_std
    noise_imag = torch.randn_like(signal.imag) * noise_std
    noise = torch.complex(noise_real, noise_imag)

    return signal + noise



M = 512
Nchirp = 220
N = 500
pad_len = N - Nchirp
f1, f2 = 0.12, 0.3


t_chirp = torch.arange(Nchirp, dtype=torch.float32)
true_if_chirp = f1 + (f2 - f1) / Nchirp * t_chirp
true_if_bins = true_if_chirp * M

alpha_true = 2 * math.pi * (f2 - f1) / Nchirp


idx_start = pad_len + 30
idx_end = N - 30
valid_range = range(idx_start, idx_end)
true_if_target = true_if_bins[30 : Nchirp - 30]

snr_levels = np.arange(-10, 60, 7)
num_mc = 10

tfrvs1 = tfrvsgab2(M=M, L=10.0, gamma_K=1e-4, q_method=2, if_method=1)
tfrvs2 = tfrvsgab2(M=M, L=10.0, gamma_K=1e-4, q_method=3, if_method=1)
tfrvs3 = tfrvsgab2(M=M, L=10.0, gamma_K=1e-4, q_method=2, if_method=2)
tfrvs4 = tfrvsgab2(M=M, L=10.0, gamma_K=1e-4, q_method=1, if_method=1)

x_clean, _ = generate_signal_2(Fs=1000.0, complex_signal=True, device='cpu')

mse_if1_q2_results = []
mse_if1_q3_results = []
mse_if2_q2_results = []
mse_if2_q3_results = []
alpha_q2_results = []
alpha_q3_results = []
alpha_q4_results = []
alpha_err_q2_results = []
alpha_err_q3_results = []
alpha_err_q4_results = []


for snr in snr_levels:
    mc_mses_if1_q2 = []
    mc_mses_if1_q3 = []
    mc_mses_if2_q2 = []
    mc_mses_if2_q3 = []
    mc_alpha_q2 = []
    mc_alpha_q3 = []
    mc_alpha_q4 = []
    mc_alpha_err_q2 = []
    mc_alpha_err_q3 = []
    mc_alpha_err_q4 = []

    for mc in range(num_mc):
        x_noisy = add_awgn(x_clean, snr)

        _, stfr1, _, qhat_map1, if_hatmap1 = tfrvs1.forward(x_noisy)
        _, stfr2, _, qhat_map2, if_hatmap2 = tfrvs2.forward(x_noisy)
        _, stfr3, _, qhat_map3, if_hatmap3 = tfrvs3.forward(x_noisy)
        _, stfr4, _, qhat_map4, if_hatmap4 = tfrvs4.forward(x_noisy)
        alpha1 = torch.imag(qhat_map1)
        alpha2 = torch.imag(qhat_map2)
        alpha4 = torch.imag(qhat_map4)

        ridge_m1 = torch.argmax(torch.abs(stfr1), dim=0)
        ridge_m2 = torch.argmax(torch.abs(stfr2), dim=0)
        ridge_m3 = torch.argmax(torch.abs(stfr3), dim=0)
        ridge_m4 = torch.argmax(torch.abs(stfr4), dim=0)

        est_if_bins1 = torch.zeros(N)
        est_if_bins2 = torch.zeros(N)
        est_if_bins3 = torch.zeros(N)
        est_if_bins4 = torch.zeros(N)

        for n in range(N):
            est_if_bins1[n] = if_hatmap1[ridge_m1[n], n] * M
            est_if_bins2[n] = if_hatmap2[ridge_m2[n], n] * M
            est_if_bins3[n] = if_hatmap3[ridge_m3[n], n] * M
            est_if_bins4[n] = if_hatmap4[ridge_m4[n], n] * M
        mse1 = torch.mean((est_if_bins1[valid_range] - true_if_target) ** 2).item()
        mse2 = torch.mean((est_if_bins2[valid_range] - true_if_target) ** 2).item()
        mse3 = torch.mean((est_if_bins3[valid_range] - true_if_target) ** 2).item()
        mse4 = torch.mean((est_if_bins4[valid_range] - true_if_target) ** 2).item()

        mc_mses_if1_q2.append(mse1)
        mc_mses_if1_q3.append(mse2)
        mc_mses_if2_q2.append(mse3)
        mc_mses_if2_q3.append(mse4)

        alpha1_ridge = torch.tensor([alpha1[ridge_m1[n], n] for n in valid_range])
        alpha2_ridge = torch.tensor([alpha2[ridge_m2[n], n] for n in valid_range])
        alpha4_ridge = torch.tensor([alpha4[ridge_m4[n], n] for n in valid_range])

        alpha1_ridge = alpha1_ridge[torch.isfinite(alpha1_ridge)]
        alpha2_ridge = alpha2_ridge[torch.isfinite(alpha2_ridge)]
        alpha4_ridge = alpha4_ridge[torch.isfinite(alpha4_ridge)]

        alpha1_mean = torch.mean(torch.abs(alpha1_ridge)).item() if len(alpha1_ridge) > 0 else 0.0
        alpha2_mean = torch.mean(torch.abs(alpha2_ridge)).item() if len(alpha2_ridge) > 0 else 0.0
        alpha4_mean = torch.mean(torch.abs(alpha4_ridge)).item() if len(alpha4_ridge) > 0 else 0.0

        mc_alpha_q2.append(alpha1_mean)
        mc_alpha_q3.append(alpha2_mean)
        mc_alpha_q4.append(alpha4_mean)
        mc_alpha_err_q2.append(abs(alpha1_mean - alpha_true))
        mc_alpha_err_q3.append(abs(alpha2_mean - alpha_true))
        mc_alpha_err_q4.append(abs(alpha4_mean - alpha_true))

    mse_if1_q2_results.append(10 * np.log10(np.mean(mc_mses_if1_q2) + 1e-12))
    mse_if1_q3_results.append(10 * np.log10(np.mean(mc_mses_if1_q3) + 1e-12))
    mse_if2_q2_results.append(10 * np.log10(np.mean(mc_mses_if2_q2) + 1e-12))
    mse_if2_q3_results.append(10 * np.log10(np.mean(mc_mses_if2_q3) + 1e-12))

    alpha_q2_results.append(np.mean(mc_alpha_q2))
    alpha_q3_results.append(np.mean(mc_alpha_q3))
    alpha_q4_results.append(np.mean(mc_alpha_q4))
    alpha_err_q2_results.append(np.mean(mc_alpha_err_q2))
    alpha_err_q3_results.append(np.mean(mc_alpha_err_q3))
    alpha_err_q4_results.append(np.mean(mc_alpha_err_q4))

    print(
        f"SNR: {snr:2d} dB | "
        f"MSE (if1, q2): {mse_if1_q2_results[-1]:.2f} dB | "
        f"MSE (if1, q3): {mse_if1_q3_results[-1]:.2f} dB | "
        f"MSE (if2, q2): {mse_if2_q2_results[-1]:.2f} dB | "
        f"MSE (if2, q3): {mse_if2_q3_results[-1]:.2f} dB"
    )

plt.figure(figsize=(9, 5))
plt.plot(snr_levels, mse_if1_q2_results, 'o-', label='phi t2 ')
plt.plot(snr_levels, mse_if2_q2_results, '^--', label='phi w2')
plt.title("Erreur d'estimation de la fréquence instantanée en fonction du  SNR")
plt.xlabel("SNR du signal (dB)")
plt.ylabel("MSE (dB)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(snr_levels, alpha_q2_results, 'o-', label='alpha estimé q_method=2')
plt.plot(snr_levels, alpha_q3_results, 's-', label='alpha estimé q_method=3')
plt.plot(snr_levels, alpha_q4_results, 'd-', label='alpha K1 q_method=1')
plt.axhline(alpha_true, color='k', linestyle='--', label='alpha théorique')
plt.title("Estimation de alpha en fonction du SNR")
plt.xlabel("SNR du signal (dB)")
plt.ylabel("alpha")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(snr_levels, alpha_err_q2_results, 'o-', label='|alpha t2 - alpha réel|')
plt.plot(snr_levels, alpha_err_q3_results, 's-', label='|alpha w2 - alpha réel|')
plt.plot(snr_levels, alpha_err_q4_results, 'd-', label='|alpha K1 - alpha réel|')
plt.title("Erreur de alpha par rapport à la pente réelle en fonction du SNR")
plt.xlabel("SNR du signal (dB)")
plt.ylabel("Erreur absolue moyenne")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
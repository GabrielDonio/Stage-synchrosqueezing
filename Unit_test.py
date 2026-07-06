import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from Transform.TfrWIn import TfrWin
from Transform.Tfrgab import Tfrgab
from Transform.STFRGab import Stfrgab
from Transform.STFRWin import Stfrwin


signal_path = "batsig.sig"
result_path = Path("/Users/macos/Documents/Stage STFT/Resultat plot")
result_path.mkdir(parents=True, exist_ok=True)

fs = 1.0 / 7e-6
M = 2048
len_win = 32
window = torch.hann_window(len_win)

ops = {
    "TfrWin (hann window 32 samples) ": TfrWin(M, hop_length=1, window=window),
    "Tfrgab": Tfrgab(M, hop_length=1),
    "STFRGab": Stfrgab(M, hop_length=1, gamma_K=1e-8),
    "STFRWin (hann window 32 samples)": Stfrwin(M, hop_length=1, window=window),
}


def rqf(x, x_at, eps=1e-12):
    return 20 * torch.log10(torch.norm(x) / (torch.norm(x - x_at) + eps))


def renyi_entropy(tfr, q=2.0, eps=1e-12):
    if q <= 0 or abs(q - 1.0) < eps:
        raise ValueError("q must be > 0 and different from 1.")
    power = torch.abs(tfr) ** 2
    prob = power / (torch.sum(power) + eps)
    return torch.log(torch.sum(prob ** q) + eps) / (1.0 - q)


def load_sig_file(path):
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                pass
    return torch.tensor(values, dtype=torch.float32)


def save_tfr_figure(tfr, title, filename, fs, x_len):
    tfr_pos = tfr[: M // 2 + 1, :]
    freqs_pos = torch.fft.rfftfreq(M, d=1 / fs)

    plt.figure(figsize=(12, 5))
    plt.imshow(
        torch.abs(tfr_pos).cpu().numpy(),
        aspect="auto",
        origin="lower",
        extent=[0, x_len / fs, freqs_pos[0].item(), freqs_pos[-1].item()],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.savefig(result_path / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def save_signal_figure(x, x_hat, title, filename):
    t = torch.arange(len(x)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t.numpy(), x.cpu().numpy(), label="Original")
    plt.plot(t.numpy(), x_hat.cpu().numpy(), label="Reconstructed", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    x = load_sig_file(signal_path)

    results = []

    for name, op in ops.items():
        out = op(x)

        if isinstance(out, tuple):
            tfr, lost = out
        else:
            tfr = out
            lost = None

        entropy = renyi_entropy(tfr).item()

        x_hat = None
        rqf_val = None
        rec_ok = hasattr(op, "rec")

        if rec_ok:
            try:
                x_hat = op.rec(tfr)
                rqf_val = rqf(x, x_hat).item()
            except Exception as e:
                print(f"[{name}] reconstruction impossible: {e}")
                rec_ok = False

        title_tfr = f"{name} | Renyi={entropy:.2f}"
        if lost is not None:
            title_tfr += f" | Lost={lost:.2f}"
        save_tfr_figure(
            tfr=tfr,
            title=title_tfr,
            filename=f"{name}_tfr.png",
            fs=fs,
            x_len=len(x),
        )

        if x_hat is not None:
            save_signal_figure(
                x=x,
                x_hat=x_hat,
                title=f"{name} reconstruction | RQF={rqf_val:.2f} dB",
                filename=f"{name}_reconstruction.png",
            )

        results.append({
            "name": name,
            "entropy": entropy,
            "lost": lost,
            "rqf": rqf_val,
            "reconstructed": rec_ok,
        })

    print("\n Summary ")
    for r in results:
        line = f"{r['name']} | Renyi={r['entropy']:.2f}"
        if r["lost"] is not None:
            line += f" | Lost_energy={r['lost']:.2f}"
        if r["rqf"] is not None:
            line += f" | RQF={r['rqf']:.2f} dB"
        print(line)

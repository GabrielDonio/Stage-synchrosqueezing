import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from Transform.TfrWIn import TfrWin
from Transform.Tfrgab import Tfrgab
from Transform.STFRGab import Stfrgab
from Transform.STFRWin import Stfrwin

from Transform.RFWin import RfWin
from Transform.RFgab import Rfgab

signal_path = "batsig.sig"
result_path = Path("/Users/macos/Documents/Stage STFT/Resultat plot")
result_path.mkdir(parents=True, exist_ok=True)

fs = 1.0 / 7e-6
M = 2048
len_win = 64
window = torch.hann_window(len_win)

ops = {
    "TfrWin (hann window " + str(len_win) + " samples) ": TfrWin(M, hop_length=1, window=window),
    "Tfrgab": Tfrgab(M, hop_length=1),
    "STFRGab": Stfrgab(M, hop_length=1, gamma_K=1e-8),
    "STFRWin (hann window " + str(len_win) + " samples)": Stfrwin(M, hop_length=1, window=window),
    "RFWin (hann window " + str(len_win) + " samples)": RfWin(M, hop_length=1, window=window),
    "RFgab": Rfgab(M, hop_length=1),
}


def rqf(x, x_at, eps=1e-12):
    return 20 * torch.log10(torch.norm(x) / (torch.norm(x - x_at) + eps))


def renyi_entropy(tfr, t=None, f=None, alpha=3.0, eps=1e-12):
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if t is None:
        t = torch.arange(tfr.shape[1], dtype=torch.float32, device=tfr.device)
    if f is None:
        f = torch.arange(tfr.shape[0], dtype=torch.float32, device=tfr.device)

    density = torch.abs(tfr) ** 2
    density = density / (integ2d(density, t, f, eps=eps) + eps)

    if abs(alpha - 1.0) < eps:
        if torch.min(density) < 0:
            raise ValueError("distribution with negative values => alpha=1 not allowed")
        return -integ2d(density * torch.log2(density + eps), t, f, eps=eps)

    return torch.log2(integ2d(density ** alpha, t, f, eps=eps) + eps) / (1.0 - alpha)


def integ2d(z, t, f, eps=1e-12):
    t = torch.as_tensor(t, dtype=torch.float32, device=z.device)
    f = torch.as_tensor(f, dtype=torch.float32, device=z.device)
    f_sorted, idx = torch.sort(f)
    z = z.index_select(0, idx)
    return torch.trapz(torch.trapz(z, t, dim=1), f_sorted, dim=0)


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
        cmap="inferno",
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
    t = torch.arange(len(x)) / fs
    title = "Original Signal"
    plt.figure(figsize=(12, 4))
    plt.plot(t.numpy(), x.cpu().numpy(), label="Original")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / "original_signal.png", dpi=200, bbox_inches="tight")
    plt.show()

    results = []

    for name, op in ops.items():
        out = op(x)

        if isinstance(out, tuple):
            tfr, lost = out
        else:
            tfr = out
            lost = None

        n_freq, n_time = tfr.shape
        t_axis = torch.arange(n_time, dtype=torch.float32, device=tfr.device)
        f_axis = torch.arange(n_freq, dtype=torch.float32, device=tfr.device)

        entropy = renyi_entropy(tfr, t=t_axis, f=f_axis, alpha=3.0).item()

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

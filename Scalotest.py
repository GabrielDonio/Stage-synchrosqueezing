import torch
from mmaxis import m_axis
from Transform.TfrvsGab2 import Tfrvsgab2
from Transform.TfrhsGab2 import Tfrthsgab2
from pathlib import Path
import matplotlib.pyplot as plt
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


signal_path = "batsig.sig"

fs = 1.0 / 7e-6
M = 2048
tfrvsgab = Tfrthsgab2(M, hop_length=1)
if __name__ == "__main__":
    x = load_sig_file(signal_path)
    x[370]=1
    t = torch.arange(len(x)) / fs
    title = "Original Signal"
    plt.figure(figsize=(12, 4))
    plt.plot(t.numpy(), x.cpu().numpy(), label="Original")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    Stfr,lost = tfrvsgab.forward(x)
    plt.imshow(torch.abs(Stfr).cpu().numpy(), aspect="auto", origin="lower")
    plt.show()

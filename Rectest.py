import torch
from Stft_rec import Stft_rec
from Sync_rec import Sync_rec
from Tfrgab import Tfrgab
from TfrWin import TfrWin
from StfrWin import Stfrwin
import matplotlib.pyplot as plt


M=512
f = 50
fs=200
t = torch.arange(0, 1, 1/fs)
x = torch.cos(2 * torch.pi * f * t**2)
#x = torch.zeros_like(t)
#x[50] = 1.0
#x[150] = 1.0
window = torch.hann_window(32)

if __name__ == "__main__":
    plt.plot(t.cpu().numpy(), x.cpu().numpy())
    plt.title("Original Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    
    tfr = Stfrwin(M, window=window)
    rtfr,lost = tfr(x)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Spectrogram (Windowed)")
    plt.colorbar()
    plt.show()
    
    sync_rec = Sync_rec(M, window=window)
    x_hat= sync_rec(rtfr)
    
    plt.plot(t.cpu().numpy(), x.cpu().numpy(), label="Original Signal")
    plt.plot(t.cpu().numpy(), x_hat.real.cpu().numpy(), label="Reconstructed Signal", linestyle='--')
    plt.title("Signal Reconstruction from STFT")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


import torch
from Reconstruction.Stft_rec import Stft_rec
from Reconstruction.Sync_rec import Sync_rec
from Reconstruction.Stft_rec_op import Stft_rechop
from Transformation.Tfrgab import Tfrgab
from Transformation.TfrWin import TfrWin
from Transformation.Tfrhop import Tfrhop
#from StfrWin import Stfrwin
from OptimizedVer.StfrWinopt import Stfrwin
import matplotlib.pyplot as plt


M=512
f = 50
fs=2000
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
    
    hop_length =32
    tfr = Tfrhop(M, window=window, hop_length=hop_length)
    rtfr = tfr(x)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Spectrogram (Windowed)")
    plt.colorbar()
    plt.show()
    
    stft_rec = Stft_rechop(M, window=window, hop_length=hop_length)
    x_hat = stft_rec(rtfr, original_N=x.shape[0])
    
    plt.plot(t.cpu().numpy(), x.cpu().numpy(), label="Original Signal")
    plt.plot(t.cpu().numpy(), x_hat.real.cpu().numpy(), label="Reconstructed Signal")
    plt.title("Signal Reconstruction from STFT")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


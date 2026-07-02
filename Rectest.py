import torch
from Reconstruction.Stft_rec import Stft_rec
from Reconstruction.Sync_rec import Sync_rec
#from Reconstruction.Stft_rec_op import Stft_rechop
from Transformation.OptimizedVer.TfrgabOpt import Tfrgab
from Transformation.TfrWin import TfrWin
from Transformation.Tfrhop import Tfrhop
#from Transformation.StfrWin import Stfrwin
from Transformation.OptimizedVer.StfrWinopt import Stfrwin
#from Transformation.StfrWin import Stfrwin
from Transformation.OptimizedVer.StfrgabOpt import Stfrgab
import matplotlib.pyplot as plt


M=128
f = 1000
fs=10000
t = torch.arange(0, torch.pi, 1/fs)
x = torch.exp(2*1j * torch.pi * f * t)
#x = torch.zeros_like(t)
#x[50] = 1.0
#x[150] = 1.0
window = torch.hann_window(128)

if __name__ == "__main__":
    plt.plot(t.cpu().numpy(), x.cpu().numpy())
    plt.title("Original Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    
    #hop_length =32
    tfr = Stfrgab(M)
    rtfr,lost= tfr(x)

    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()
   
    stft_rec = Sync_rec(M, window=window,isgab=True)
    x_hat = stft_rec(rtfr)
    
    plt.plot(t.cpu().numpy(), x.cpu().numpy(), label="Original Signal")
    plt.plot(t.cpu().numpy(), x_hat.real.cpu().numpy(), label="Reconstructed Signal")
    plt.title("Signal Reconstruction from STFT")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


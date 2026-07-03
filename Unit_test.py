import torch
import matplotlib.pyplot as plt

from Transform.Tfrgab import Tfrgab
from Transform.TfrWIn import TfrWin
from Transform.RFgab import Rfgab
from Transform.RFWin import RfWin
from Transform.STFRGab import Stfrgab as STfrgab
from Transform.STFRWin import Stfrwin as STfrwin
from Transformation.Tfrscalo import Tfrscalo
from Transformation.OptimizedVer.StfrgabOpt import Stfrgab
from Transformation.OptimizedVer.StfrWinopt import Stfrwin
from Transformation.OptimizedVer.StfrgabhopOpt import Stfrgabhop


M=512
f = 50
fs=1000
t = torch.arange(0,torch.pi, 1/fs)
x = torch.cos(2 * torch.pi * f * t**2) + torch.cos(2*2 * torch.pi * f * t**2)

x[250] = 1.0
x[700] = 1.0

hann_window = torch.hann_window(32)

if __name__ == "__main__":
    tfr = STfrwin(M, window=hann_window, hop_length=1)
    tfr_result,lost= tfr.forward(x)


    plt.imshow(torch.abs(tfr_result), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('TFR Window')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    
    x_est = tfr.rec(tfr_result)
    print("Reconstruction error:", torch.norm(x - x_est).item())
    plt.plot(t.cpu().numpy(), x.cpu().numpy())
    plt.plot(t.cpu().numpy(), x_est.cpu().numpy(), alpha=0.7)
    plt.title('reconstructed signal')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    
    
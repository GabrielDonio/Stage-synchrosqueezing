import torch
import matplotlib.pyplot as plt
from Tfrrgab import Tfrrgab
from Tfrgab import Tfrgab
from Tfrgabhop import Tfrgabhop
from Tfrscalo import Tfrscalo
from Stfr import Stfrgab
from StfrWin import Stfrwin
from TfrWin import TfrWin


M=512
f = 50
fs=200
t = torch.arange(0, 1, 1/fs)
#x = torch.cos(2 * torch.pi * f * t**2)
x = torch.zeros_like(t)
x[50] = 1.0
x[150] = 1.0

hann_window = torch.hann_window(32)

if __name__ == "__main__":
    hop_length = 32
    tfrgab_hop = Tfrgabhop(M, hop_length)
    tfrgab = Tfrgabhop(M, hop_length)
    rtfr = tfrgab(x)
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Spectrogramme avec hop_length (Gabor)")
    plt.colorbar()
    plt.show()
    
    tfrgab = Tfrgab(M)
    rtfr = tfrgab(x)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Spectrogram (Gabor)")
    plt.colorbar()
    plt.show()
    
    tfrrgab = Tfrrgab(M)
    rtfr, lost = tfrrgab(x)
    print(lost)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Spectrogramme Réassigné (Gabor)")
    plt.colorbar()
    plt.show()
    
    stfrgab = Stfrgab(M)
    rtfr, lost = stfrgab(x)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Synchrosqueezed (Gabor)")
    plt.colorbar()
    plt.show()
    
    tfrwin = TfrWin(M, hann_window)
    rtfr = tfrwin(x)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Spectrogramme avec fenetre aléatoire")
    plt.colorbar()
    plt.show()
    
    stfrwin = Stfrwin(M, hann_window)
    rtfr, lost = stfrwin(x)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Synchrosqueezed (hann window)")
    plt.colorbar()    
    plt.show()
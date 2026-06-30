import torch
import matplotlib.pyplot as plt
from Tfrrgab import Tfrrgab
from Tfrgab import Tfrgab
from Tfrgabhop import Tfrgabhop
from Tfrscalo import Tfrscalo
from Stfr import Stfrgab


M=512
f = 50
fs=250
t = torch.arange(0, 1, 1/fs)
x = torch.cos(2 * torch.pi * f * t**2)


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
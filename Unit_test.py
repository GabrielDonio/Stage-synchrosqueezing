import torch
import matplotlib.pyplot as plt
#from Tfrrgab import Tfrrgab
from Transformation.OptimizedVer.TfrrgabOpt import Tfrrgab
#from Tfrgab import Tfrgab
from Transformation.OptimizedVer.TfrgabOpt import Tfrgab 
#from Tfrgabhop import Tfrgabhop
#from OptimizedVer.TfrgabhopOpt import Tfrgabhop
from Transformation.Tfrhop import Tfrhop
from Transformation.Tfrscalo import Tfrscalo
#from Stfrgab import Stfrgab
from Transformation.OptimizedVer.StfrgabOpt import Stfrgab
from Transformation.OptimizedVer.StfrWinopt import Stfrwin
#from TfrWin import TfrWin
from Transformation.OptimizedVer.TfrWinOpt import TfrWin#
from Transformation.OptimizedVer.TfrrgabOpt import Tfrrgab
from Transformation.OptimizedVer.StfrgabhopOpt import Stfrgabhop


M=512
f = 50
fs=1000
t = torch.arange(0,torch.pi, 1/fs)
x = torch.cos(2 * torch.pi * f * t**2)
#x = torch.zeros_like(t)
#x[10] = 1.0
#x[500] = 1.0

hann_window = torch.hann_window(32)

if __name__ == "__main__":
    hop_length = 32
    tfrhop = Tfrhop(M, hann_window, hop_length)
    rtfr = tfrhop(x)
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
    
    stfrgabhop = Stfrgabhop(M, hop_length)
    rtfr, lost = stfrgabhop(x)
    
    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Synchrosqueezed (Gabor with hop_length)")
    plt.colorbar()
    plt.show()
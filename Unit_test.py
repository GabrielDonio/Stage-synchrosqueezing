import torch
import torchaudio
import matplotlib.pyplot as plt

from Transform.Tfrgab import Tfrgab
from Transform.TfrWIn import TfrWin
from Transform.RFgab import Rfgab
from Transform.RFWin import RfWin
from Transform.STFRGab import Stfrgab as STfrgab
from Transform.STFRWin import Stfrwin as STfrwin
from Transform.Scalo import Scalo

#signal_path = "/Users/macos/Documents/Audio python/Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57.wav"

#signal, fs = torchaudio.load(signal_path)
#signal = signal[0]  


M=1024

fs = 300
f_base = 100
f_mod =3
mod_amplitude = 10
amplitude = 1
T = 0.1 
Ts = 1/300
w0 = 2 * torch.pi * 50     

t = torch.arange(0, 1, 1/fs)

phase1 = 2 * torch.pi * ( f_base * t- (mod_amplitude / (2 * torch.pi * f_mod)) * torch.cos(2 * torch.pi * f_mod * t))
x = amplitude * torch.exp(1j * phase1) 
eps = torch.randn_like(x) 
x += eps
#x[250] = 1.0
#x[700] = 1.0

hann_window = torch.hann_window(16)
tfr = Rfgab(M, hop_length=1)  

if __name__ == "__main__":

    tfr,lost=tfr.forward(x)
    
    plt.imshow(torch.abs(tfr).numpy(), aspect='auto', origin='lower')
    plt.colorbar(label='magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    
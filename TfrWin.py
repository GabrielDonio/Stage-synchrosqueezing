import torch
import matplotlib.pyplot as plt


class Tfrfen(torch.nn.Module):
    def __init__(self, M, window, eps=1e-6):
        super().__init__()
        self.M = M
        self.window = torch.as_tensor(window, dtype=torch.float32, device=window.device).reshape(-1)  # window
        self.len_win = window.shape[0]
        self.eps = eps

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        N = x.shape[0]
        device = x.device
        tfr = torch.zeros((self.M, N), dtype=torch.complex64, device=device)

        mm = torch.arange(0, self.M, device=device)

        center = self.len_win // 2
        left = center
        right = self.len_win - center - 1

        for n in range(N):
            k_min = min(n, left)
            k_max = min(N - 1 - n, right)

            k = torch.arange(-k_min, k_max + 1, device=device)
            w = self.window[center - k_min : center + k_max + 1]
            x_slice = x[n + k]

            for m in range(self.M):
                exp_B_mm_k = torch.exp(-1j * 2 * torch.pi * mm[m] * k / self.M)
                tfr[m, n] = torch.sum(x_slice * w * exp_B_mm_k)

        return tfr

if __name__ == "__main__":
    M = 512
    window = torch.hann_window(128)
    tfrfen = Tfrfen(M, window)
    fs = 300
    f = 50
    t = torch.arange(0, 1, 1/fs)
    x = torch.cos(2 * torch.pi * f * t**2)
    rtfr = tfrfen(x)

    plt.imshow(torch.log1p(rtfr.abs()).cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Spectrogramme avec fenetre aléatoire")
    plt.colorbar()
    plt.show()


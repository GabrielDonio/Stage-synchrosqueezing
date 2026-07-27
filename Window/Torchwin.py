import torch 


def hann_window(window_length: int, periodic: bool = True) -> torch.Tensor:
    return torch.hann_window(window_length, periodic=periodic)

def hamming_window(window_length: int, periodic: bool = True) -> torch.Tensor:
    return torch.hamming_window(window_length, periodic=periodic)
def blackman_window(window_length: int, periodic: bool = True) -> torch.Tensor:
    return torch.blackman_window(window_length, periodic=periodic)
def bartlett_window(window_length: int, periodic: bool = True) -> torch.Tensor:
    return torch.bartlett_window(window_length, periodic=periodic)
def kaiser_window(window_length: int, periodic: bool = True, beta: float = 14.0) -> torch.Tensor:
    return torch.kaiser_window(window_length, periodic=periodic, beta=beta)

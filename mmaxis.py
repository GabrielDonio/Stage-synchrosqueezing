import torch

def m_axis(M, i=None, method=1, device=None):
    """
    Compute the frequency bins in range {0} U [1, M/2] U [-M/2+1, -1]

    Returns:
        torch.Tensor: frequency axis
    """
    mm = torch.arange(1, M + 1, device=device)

    Mh = M // 2

    if method == 1:
        I1 = mm <= (Mh + 1)
        I2 = mm > (Mh + 1)
        mm = mm.clone()
        mm[I1] = mm[I1] - 1
        mm[I2] = -(M + 1) + mm[I2]
    else:
        I1 = mm <= Mh
        I2 = mm > Mh
        mm = mm.clone()
        mm[I1] = mm[I1] - 1
        mm[I2] = -M + mm[I2]

    if i is not None:
        mm = mm[i]

    return mm

def a_axis(M, as_range, method=0, i=None, device=None, dtype=torch.float32):
    """
    Compute the scale values in range [as_range[0], as_range[1]]

    Returns:
        torch.Tensor: scale axis
    """
    a0, a1 = as_range

    if method == 0:
        as_axis = torch.logspace(
            torch.log10(torch.tensor(a0, device=device, dtype=dtype)),
            torch.log10(torch.tensor(a1, device=device, dtype=dtype)),
            M,
            device=device,
            dtype=dtype,
        )
    elif method == 1:
        as_axis = 1.0 / torch.linspace(
            a0, a1, M, device=device, dtype=dtype
        )
    elif method == 2:
        as_axis = torch.linspace(
            a0, a1, M, device=device, dtype=dtype
        )
    else:
        raise ValueError("method must be 0, 1, or 2")

    if i is not None:
        as_axis = as_axis[i]

    return as_axis
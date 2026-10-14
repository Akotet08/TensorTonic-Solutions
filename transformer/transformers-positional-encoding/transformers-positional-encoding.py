import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    base = 10000
    pos = np.arange(seq_length).reshape(-1, 1)
    denom = base ** (2 * (np.arange(d_model) // 2) / d_model)

    mat = pos / denom.reshape(1, -1)

    mat[:, 0::2] = np.sin(mat[:, 0::2])
    mat[:, 1::2] = np.cos(mat[:, 1::2])

    return mat
    
    
    
    
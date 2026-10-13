import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """

    pos = np.arange(seq_len, dtype=float).reshape(-1, 1)
    denom = base ** (2 * (np.arange(d_model, dtype=float) // 2) / d_model).reshape(1, -1)
   
    mat = pos / denom

    mat[:, 0::2] = np.sin(mat[:, 0::2])
    mat[:, 1::2] = np.cos(mat[:, 1::2])

    return mat


    
    
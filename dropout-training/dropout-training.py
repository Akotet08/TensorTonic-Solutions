import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.array(x) * (1/(1-p))

    mask = rng.random(x.shape) if rng else np.random.random(x.shape)
    mask = mask > p

    x[~mask] = 0

    return x, mask.astype(int) * (1/(1-p))
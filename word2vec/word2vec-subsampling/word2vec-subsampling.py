import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    f_w = counts / counts.sum()
    ones = torch.ones_like(f_w)

    p_keep = torch.minimum(ones, torch.sqrt(t / f_w))

    return p_keep 

    
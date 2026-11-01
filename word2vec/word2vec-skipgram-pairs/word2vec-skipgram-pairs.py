import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    acc = []
    
    for j in range(len(token_ids)):
        start = max(0, j - window)
        end = min(len(token_ids) - 1, j + window)

        for p in range(start,  end + 1 ):
            if p != j:
                acc.append([token_ids[j], token_ids[p]])


    return torch.tensor(acc, dtype=torch.int64).reshape(-1, 2)
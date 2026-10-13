import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    
    lengths = [len(seq) for seq in seqs] if len(seqs) else [0]
    N, L = len(seqs), max_len if max_len != None else max(lengths)
    
    padded = np.full((N, L), pad_value)

    for i, seq in enumerate(seqs):
        l = min(len(seq), L)
        padded[i, :l] = seq[:l]
        
    return padded
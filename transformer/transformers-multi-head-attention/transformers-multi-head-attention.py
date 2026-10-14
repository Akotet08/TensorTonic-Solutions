import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    def softmax(x, axis=-1):
        kw = dict(axis=axis, keepdims=True)
        e_x = np.exp(x - np.max(x, **kw))
        
        return e_x / np.sum(e_x, **kw)

    
    def head(q, k, v):
        dk = q.shape[2]
        
        sft = softmax(
            np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(dk), 
            axis= -1
        )

        return np.matmul(sft, v)

    _q = np.matmul(Q, W_q)
    _k = np.matmul(K, W_k)
    _v = np.matmul(V, W_v)

    heads = []
    _, _, d_model = _q.shape
    d = int(d_model / num_heads)
    for h in range(num_heads):
        q = _q[:, :, h*(d): (h+1)*(d)]
        k = _k[:, :, h*(d): (h+1)*(d)]
        v = _v[:, :, h*(d): (h+1)*(d)]

        heads.append(head(q, k, v))

    heads_con = np.concatenate(heads, axis=-1)
    return np.matmul(heads_con, W_o)
        
    
    
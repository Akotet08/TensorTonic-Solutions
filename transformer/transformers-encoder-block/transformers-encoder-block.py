import numpy as np

def layer_norm(x, gamma, beta, eps = 1e-6):
    mu = np.mean(x, axis = -1, keepdims=True)
    var = np.var(x, axis  = -1, keepdims=True)

    return gamma * ((x - mu) / (np.sqrt(var + eps))) + beta

def feed_forward(x, W1, b1, W2, b2):
    z = x @ W1 + b1
    mask = z < 0
    z[mask] = 0

    return z @ W2 + b2

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    
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
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    Q, K, V = x, x, x 
    z = layer_norm(x + multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    return layer_norm(z + feed_forward(z, W1, b1, W2, b2), gamma2, beta2)
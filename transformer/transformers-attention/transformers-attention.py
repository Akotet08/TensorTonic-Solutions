import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """

    _, _, dk = K.shape

    sft = torch.softmax(torch.matmul(Q, torch.transpose(K, -1, -2)) / torch.sqrt(torch.tensor(dk)), dim=-1) 
    
    return torch.matmul(sft, V)
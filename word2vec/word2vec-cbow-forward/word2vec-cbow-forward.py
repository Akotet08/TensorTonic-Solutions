import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int,
                 W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 CBOW cross-entropy loss.
    """
    M = context_ids.shape[0]
    h = torch.sum(W_in[context_ids, :], dim=0) /  M

    sft = F.softmax(W_out @ h)

    return -torch.log(sft[target_id])
    
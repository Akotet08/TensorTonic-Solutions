import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 SGNS loss.
    """

    pos_score = torch.sum(center_vec * pos_vec)
    neg_score = torch.sum(-center_vec * neg_vecs, dim=-1)

    return -torch.log(F.sigmoid(pos_score)) - torch.sum(torch.log(F.sigmoid(neg_score)))
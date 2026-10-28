import torch
import torch


def _forward(inputs, weights, biases):
    outputs = [inputs]
    for w, b in zip(weights, biases):
        outputs.append(torch.tanh(outputs[-1] @ w.T + b))
    return outputs


def _backward(grad, outputs, weights):
    dw_list, db_list = [], []

    for d in range(len(weights) - 1, -1, -1):
        dz = grad * (1 - outputs[d + 1].square())
        dw_list.append(dz.T @ outputs[d])
        db_list.append(dz.sum(dim=0))
        grad = dz @ weights[d]

    return dw_list[::-1], db_list[::-1]
    
def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    ws = [w.detach().clone() for w in weights]
    bs = [b.detach().clone() for b in biases]
    targets_column = targets.unsqueeze(-1)
    losses = []

    for _ in range(steps):
        outputs = _forward(inputs, ws, bs)
        error = outputs[-1] - targets_column
        losses.append(error.square().sum())

        dw_list, db_list = _backward(2 * error, outputs, ws)

        for w, dw in zip(ws, dw_list):
            w -= learning_rate * dw
        for b, db in zip(bs, db_list):
            b -= learning_rate * db

    final_pred = _forward(inputs, ws, bs)[-1].squeeze(-1)
    final_loss = (final_pred - targets).square().sum()

    return final_pred, final_loss, ws, bs, losses

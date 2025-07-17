from typing import Callable, Tuple
from ..config import Parameters
import torch

def l2_loss(preds: torch.Tensor, y_n: torch.Tensor, y_np1: torch.Tensor) -> torch.Tensor:
    i_n, v_n, i_np1, v_np1 = preds
    i0, v0 = y_n[:, :1], y_n[:, 1:]
    i1, v1 = y_np1[:, :1], y_np1[:, 1:]
    return (
        (i_n - i0).pow(2).sum()
        + (v_n - v0).pow(2).sum()
        + (i_np1 - i1).pow(2).sum()
        + (v_np1 - v1).pow(2).sum()
    )
    



def l1_loss(preds: torch.Tensor, y_n: torch.Tensor, y_np1: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 regularization loss for the parameters.
    This is used to encourage sparsity in the parameter estimates.
    """
    i_n, v_n, i_np1, v_np1 = preds
    i0, v0 = y_n[:, :1], y_n[:, 1:]
    i1, v1 = y_np1[:, :1], y_np1[:, 1:]

    # L1 regularization on the predictions
    return (
        torch.abs(i_n - i0).sum()
        + torch.abs(v_n - v0).sum()
        + torch.abs(i_np1 - i1).sum()
        + torch.abs(v_np1 - v1).sum()
    )


def weighted_l2_loss(
    preds: torch.Tensor,
    y_n: torch.Tensor,
    y_np1: torch.Tensor,
    weights: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    i_n, v_n, i_np1, v_np1 = preds
    i0, v0 = y_n[:, :1], y_n[:, 1:]
    i1, v1 = y_np1[:, :1], y_np1[:, 1:]
    weights_i, weights_v = weights

    return (
        (2 * weights_i * (i_n - i0).pow(2)).sum()
        + (2 * weights_v * (v_n - v0).pow(2)).sum()
        + (2 * weights_i * (i_np1 - i1).pow(2)).sum()
        + (2 * weights_v * (v_np1 - v1).pow(2)).sum()
    )

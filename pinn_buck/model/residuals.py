import torch
from typing import Callable, Dict, Tuple, Union

ResidualFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# define decorator that applies a residual function

def residual_function_class(residual_func: ResidualFunc) -> ResidualFunc:
    def wrapper(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return residual_func(pred, target)
    return wrapper


@residual_function_class
def basic_residual(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pred - target

@residual_function_class
def first_order_derivative_residual(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # both forward and backward predictions are necessary, check the dimensions of the torch tensors
    # first check the number of dimensions: (2, B, T, 2)
    if pred.dim() != 4 or target.dim() != 4:
        raise ValueError(f"Both pred and target must be 4-dimensional tensors: pred shape {pred.shape}, target shape {target.shape}")
    if pred.shape[0] != 2 or target.shape[0] != 2:
        raise ValueError(f"Both pred and target must have 2 elements in the first dimension: pred shape {pred.shape}, target shape {target.shape}")
    pred_fwd, pred_bck = pred
    target_fwd, target_bck = target
    return (pred_fwd - pred_bck) - (target_fwd - target_bck)
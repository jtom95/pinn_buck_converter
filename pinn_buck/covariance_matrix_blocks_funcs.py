"""
This module provides utilities for constructing and manipulating covariance matrices
used in the noise modeling.
"""

import torch
from typing import Union, Iterable, Callable


def covariance_matrix_on_standard_residuals(data_covariance: torch.Tensor, jac_fwd: torch.Tensor, jac_bck: torch.Tensor, dtype: torch.dtype = torch.float32) -> dict:
    """
    Estimate the Sigma matrix for the 2x2 blocks.

    Args:
        sigma_x (torch.Tensor): 2x2 matrix representing noise covariance.
        jac_fwd (torch.Tensor): 2x2 jacacobian matrix of the forward prediction function.
        jac_bck (torch.Tensor): 2x2 jacacobian matrix of the backward prediction function.

    Returns:
        dict: Dictionary containing the 2x2 Sigma blocks:
            - "fwfw": Forward-forward block.
            - "fwbw": Forward-backward block.
            - "bwfw": Backward-forward block.
            - "bwbw": Backward-backward block.
    """
    if data_covariance.shape != (2, 2):
        raise ValueError("data_covariance must be a 2x2 matrix.")
    if jac_fwd.shape != (2, 2):
        raise ValueError("jac must be a 2x2 matrix.")
    if jac_bck.shape != (2, 2):
        raise ValueError("jac_inv must be a 2x2 matrix.")
    
    # cast to the correct dtype
    data_covariance = data_covariance.to(dtype)
    jac_fwd = jac_fwd.to(dtype)
    jac_bck = jac_bck.to(dtype)


    return {
        "fwfw": jac_fwd @ data_covariance @ jac_fwd.T + data_covariance,
        "fwbw": -jac_fwd @ data_covariance - data_covariance @ jac_bck.T,
        "bwfw": -jac_bck @ data_covariance - data_covariance @ jac_fwd.T,
        "bwbw": jac_bck @ data_covariance @ jac_bck.T + data_covariance,
    }


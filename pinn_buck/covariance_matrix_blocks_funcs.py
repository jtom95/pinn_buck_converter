"""
This module provides utilities for constructing and manipulating covariance matrices
used in the noise modeling.
"""

import torch
from typing import Union, Iterable, Callable


def covariance_matrix_on_standard_residuals(data_covariance: torch.Tensor, jac: torch.Tensor) -> dict:
    """
    Estimate the Sigma matrix for the 2x2 blocks.

    Args:
        sigma_x (torch.Tensor): 2x2 matrix representing noise covariance.
        jac (torch.Tensor): 2x2 jacacobian matrix of the prediction function.

    Returns:
        dict: Dictionary containing the 2x2 Sigma blocks:
            - "fwfw": Forward-forward block.
            - "fwbw": Forward-backward block.
            - "bwfw": Backward-forward block.
            - "bwbw": Backward-backward block.
    """
    if data_covariance.shape != (2, 2):
        raise ValueError("data_covariance must be a 2x2 matrix.")
    if jac.shape != (2, 2):
        raise ValueError("jac must be a 2x2 matrix.")

    jac_inv: torch.Tensor = torch.linalg.inv(jac)

    return {
        "fwfw": jac @ data_covariance @ jac.T + data_covariance,
        "fwbw": -jac @ data_covariance - data_covariance @ jac_inv.T,
        "bwfw": -jac_inv @ data_covariance - data_covariance @ jac.T,
        "bwbw": jac_inv @ data_covariance @ jac_inv.T + data_covariance,
    }


"""
This module provides utilities for constructing and manipulating covariance matrices
used in the noise modeling.
"""

import torch
from typing import Union, Iterable, Callable
from .covariance_matrix_config import (
    ResidualCovariance, ResidualCovarianceMatrixFunc, DataCovariance, 
    ResidualCovarianceFunctionFactory,
    _parse_data_noise, _ensure_positive_definite
    )


def covariance_matrix_on_residuals_d1(
    data_covariance: torch.Tensor,
    jac_fwd: torch.Tensor,
    jac_bck: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Covariance for Δ-residuals built from forward/backward predictions:
        r = (fwd - bck) - (target_fwd - target_bck)
    Under independent measurement noise with covariance Σ_x at each side,
    a linearization gives:
        Cov[r] ≈ (J_f + I) Σ_x (J_f + I)ᵀ + (J_b + I) Σ_x (J_b + I)ᵀ

    Shapes:
      data_covariance: (..., 2, 2) or (2, 2)
      jac_fwd, jac_bck: (..., 2, 2) or (2, 2)
    """
    dc = data_covariance.to(dtype)
    jf = jac_fwd.to(dtype)
    jb = jac_bck.to(dtype)

    # Broadcast to common batch if provided
    I = torch.eye(2, dtype=dtype, device=dc.device).expand(dc.shape[:-2] + (2, 2))
    jf_p = jf + I
    jb_p = jb + I

    Sigma = jf_p @ dc @ jf_p.mT + jb_p @ dc @ jb_p.mT
    return Sigma


def covariance_matrix_on_residuals_d2(
    data_covariance: torch.Tensor,
    jac_fwd: torch.Tensor,
    jac_bck: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Covariance for Δ-residuals built from forward/backward predictions:
        r = (fwd - bck) - (target_fwd - target_bck)
    Under independent measurement noise with covariance Σ_x at each side,
    a linearization gives:
        Cov[r] ≈ (J_f + I) Σ_x (J_f + I)ᵀ + (J_b + I) Σ_x (J_b + I)ᵀ

    Shapes:
      data_covariance: (..., 2, 2) or (2, 2)
      jac_fwd, jac_bck: (..., 2, 2) or (2, 2)
    """
    dc = data_covariance.to(dtype)
    jf = jac_fwd.to(dtype)
    jb = jac_bck.to(dtype)

    # Broadcast to common batch if provided
    I = torch.eye(2, dtype=dtype, device=dc.device).expand(dc.shape[:-2] + (2, 2))
    jf_p = jf - I
    jb_p = jb - I

    Sigma = jf_p @ dc @ jf_p.mT + jb_p @ dc @ jb_p.mT
    return Sigma / 4


def covariance_matrix_on_basic_residuals(
    data_covariance: torch.Tensor,
    jac: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Covariance for basic forward-only residuals r = f(y_k) - y_{k+1} (linearized):
        Cov[r] ≈ J Σ_x Jᵀ + Σ_x

    Shapes:
      data_covariance: (..., 2, 2) or (2, 2)
      jac:             (..., 2, 2) or (2, 2)
    """
    dc = data_covariance.to(dtype)
    J = jac.to(dtype)
    return J @ dc @ J.mT + dc


####


# Function to generate residual covariance matrix from the block builders defined above.
def generate_residual_covariance_matrix(
    data_covariance: Union[float, torch.Tensor, Iterable],
    residual_covariance_func: ResidualCovarianceMatrixFunc,
    damp: float = 1e-8,
    **kwargs
) -> ResidualCovariance:

    # 1) Parse the data noise input and ensure it is in the correct form
    data_covariance_matrix: DataCovariance = _parse_data_noise(data_covariance)

    # 2) Create the residual covariance function
    residual_covariance_func = ResidualCovarianceFunctionFactory.wrap_residual_covariance(
        residual_covariance_func
    )

    # 3) Assemble the covariance matrix blocks
    residual_covariance_matrix: ResidualCovariance = residual_covariance_func(
        data_covariance=data_covariance_matrix, **kwargs
    )   

    return _ensure_positive_definite(residual_covariance_matrix, damp=damp)


def chol(mat: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """return L where LLᵀ = mat (add jitter if needed)"""
    # if the numdims of mat is 2, we can use the cholesky decomposition directly
    if mat.dim() == 2:
        mat = mat + eps * torch.eye(mat.size(0), device=mat.device)
        L = torch.linalg.cholesky(mat)
        return L  # same as L⁻ᵀ · L⁻¹
    if mat.dim() == 3:
        # we apply chol to each 2x2 matrix in the batch
        return torch.stack([chol(mat[i], eps=eps) for i in range(mat.size(0))], dim=0)
    raise ValueError("Input tensor must be 2D or 3D.")

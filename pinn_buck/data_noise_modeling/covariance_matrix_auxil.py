from typing import Union, Iterable, Callable

import torch

from .covariance_matrix_configs import (
    CovarianceMatrixBuilderFactory,
    CovarianceMatrixBuilder,
    SigmaBlocksDict,
    _ensure_positive_definite,
    _parse_data_noise,
)


#####
# Function to generate residual covariance matrix from the block builders defined above.
def generate_residual_covariance_matrix(
    data_covariance: Union[float, torch.Tensor, Iterable],
    residual_covariance_block_func: Callable,
    include_diag_terms: bool = True,
    damp: float = 1e-8,
    **kwargs
) -> torch.Tensor:

    # 1) Wrap the user's function with the ABC factory – this *only* checks
    #    “does the function return a dict with the four expected keys?”
    block_builder: CovarianceMatrixBuilder = (
        CovarianceMatrixBuilderFactory.wrap_covariance_matrix_builder(
            residual_covariance_block_func
        )
    )

    # 2) Parse the data noise input and ensure it is in the correct form
    data_covariance_matrix = _parse_data_noise(data_covariance)

    # 3) Assemble the covariance matrix blocks
    blocks: SigmaBlocksDict = block_builder(data_covariance_matrix, **kwargs)
    Sigma = torch.zeros((4, 4), device=data_covariance_matrix.device)
    Sigma[:2, :2] = blocks["fwfw"]
    Sigma[2:, 2:] = blocks["bwbw"]
    if include_diag_terms:
        Sigma[:2, 2:] = blocks["fwbw"]
        Sigma[2:, :2] = blocks["bwfw"]

    # 4) Ensure the covariance matrix is positive definite
    Sigma = _ensure_positive_definite(Sigma, damp=damp)

    return Sigma


## Cholesky decomposition of the covariance matrix
def chol_inv(mat: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """return (LLᵀ)⁻¹ᐟ² = L⁻ᵀ   where LLᵀ = mat (add jitter if needed)"""
    if mat.dim() == 2:
        mat = mat + eps * torch.eye(mat.size(0), device=mat.device)
        L = torch.linalg.cholesky(mat)
        return torch.cholesky_inverse(L)  # same as L⁻ᵀ · L⁻¹
    if mat.dim() == 3:
        # we apply chol_inv to each 2x2 matrix in the batch
        return torch.stack([
            chol_inv(mat[i], eps=eps) for i in range(mat.size(0))
        ], dim=0)
    raise ValueError("Input tensor must be 2D or 3D.")


def chol(mat: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """return L where LLᵀ = mat (add jitter if needed)"""
    # if the numdims of mat is 2, we can use the cholesky decomposition directly
    if mat.dim() == 2:
        mat = mat + eps * torch.eye(mat.size(0), device=mat.device)
        L = torch.linalg.cholesky(mat)
        return L  # same as L⁻ᵀ · L⁻¹
    if mat.dim() == 3:
        # we apply chol to each 2x2 matrix in the batch
        return torch.stack([
            chol(mat[i], eps=eps) for i in range(mat.size(0))
        ], dim=0)
    raise ValueError("Input tensor must be 2D or 3D.")
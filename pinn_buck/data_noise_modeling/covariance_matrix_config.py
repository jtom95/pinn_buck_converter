from typing import Union, Iterable, Callable
from abc import ABC, abstractmethod
import torch

ResidualCovariance = torch.Tensor
DataCovariance = torch.Tensor

class ResidualCovarianceMatrixFunc(ABC):
    @abstractmethod
    def __call__(
        self,
        data_covariance: DataCovariance,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the residual covariance matrix.
        """
        pass


class ResidualCovarianceFunctionFactory:
    @staticmethod
    def wrap_residual_covariance(residual_covariance_func: Callable) -> ResidualCovarianceMatrixFunc:
        """
        Dynamically wrap a normal function into a ResidualCovarianceMatrixFunc implementation.
        """

        class LikelihoodLossFunctionImpl(ResidualCovarianceMatrixFunc):
            def __call__(
                self,
                data_covariance: DataCovariance,
                **kwargs,
            ) -> torch.Tensor:
                return residual_covariance_func(
                    data_covariance=data_covariance,
                    **kwargs,
                )

        return LikelihoodLossFunctionImpl()


def _parse_data_noise(data_noise: Union[float, torch.Tensor, Iterable], **kwargs) -> DataCovariance:
    if isinstance(data_noise, float):
        a, b = data_noise, data_noise
    elif isinstance(data_noise, torch.Tensor):
        if data_noise.shape == (2,):
            a, b = data_noise[0], data_noise[1]
        elif data_noise.shape == (2, 2):
            a, b = data_noise[0, 0], data_noise[1, 1]
        else:
            raise ValueError("If data_noise is a tensor, it must be 2x2.")
    elif isinstance(data_noise, Iterable):
        data_noise = list(data_noise)
        if len(data_noise) != 2:
            raise ValueError("If data_noise is iterable, it must be of length 2.")
        a, b = data_noise[0], data_noise[1]
    else:
        raise TypeError("data_noise must be float, 2-tensor, or iterable of length 2.")
    data_2x2_covariance_matrix = torch.diag(torch.tensor([a, b], dtype=torch.float32))
    return data_2x2_covariance_matrix


def _ensure_positive_definite(
    Sigma: ResidualCovariance,
    *,
    damp: float = 1e-8,
    symmetrize: bool = True,
) -> ResidualCovariance:
    """
    Make Sigma safely SPD for downstream Cholesky.
    - Symmetrize to (Σ + Σᵀ)/2 if requested.
    - Add jitter on the last two dims.
    Supports 2D or batched (..., 2, 2).
    """
    if symmetrize:
        Sigma = 0.5 * (Sigma + Sigma.mT)
    eye = torch.eye(Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype)
    return Sigma + damp * eye

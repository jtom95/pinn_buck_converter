import torch
from abc import ABC, abstractmethod
from typing import TypedDict, Iterable, Union
from functools import wraps


class SigmaBlocksDict(TypedDict):
    fwfw: torch.Tensor
    fwbw: torch.Tensor
    bwfw: torch.Tensor
    bwbw: torch.Tensor


class CovarianceMatrixBuilder(ABC):
    @abstractmethod
    def __call__(self, sigma_x: torch.Tensor, **kwargs) -> SigmaBlocksDict:
        """
        Compute the covariance matrix blocks.
        Args:
            sigma_x (torch.Tensor): 2x2 noise covariance.
            **kwargs: Additional keyword arguments.
        Returns:
            SigmaBlocksDict: Dictionary containing the covariance matrix blocks:
                - "fwfw": Forward-forward block.
                - "fwbw": Forward-backward block.
                - "bwfw": Backward-forward block.
                - "bwbw": Backward-backward block.
        """
        pass

class CovarianceMatrixBuilderFactory:
    @staticmethod
    def wrap_covariance_matrix_builder(func: callable) -> CovarianceMatrixBuilder:
        """
        Create a CovarianceMatrixBuilderBase instance from a function.
        Args:
            func (callable): Function to be wrapped.
        Returns:
            CovarianceMatrixBuilderBase: An instance of a class that implements CovarianceMatrixBuilderBase.
        """
        class CovarianceMatrixBuilderImpl(CovarianceMatrixBuilder):
            def __init__(self, func: callable):
                self.func = func

            def __call__(
                self, data_2x2_covariance_matrix: torch.Tensor, **kwargs
            ) -> Union[SigmaBlocksDict, torch.Tensor]:
                output = self.func(data_2x2_covariance_matrix, **kwargs)
                # check that output is a dictionary with the expected keys
                if isinstance(output, torch.Tensor):
                    # check that it is a 2x2 matrix
                    if output.shape != (2, 2):
                        raise ValueError("Output must be a 2x2 tensor.")
                    return output
                
                if not isinstance(output, dict) or not all(key in output for key in ["fwfw", "fwbw", "bwfw", "bwbw"]):
                    raise ValueError("Output must be a dictionary with keys: 'fwfw', 'fwbw', 'bwfw', 'bwbw'")
                return SigmaBlocksDict(
                    fwfw=output["fwfw"],
                    fwbw=output["fwbw"],
                    bwfw=output["bwfw"],
                    bwbw=output["bwbw"]
                )  
        return CovarianceMatrixBuilderImpl(func)

def _parse_data_noise(data_noise: Union[float, torch.Tensor, Iterable], **kwargs) -> torch.Tensor:
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
        

def _ensure_positive_definite(Sigma: torch.Tensor, damp: float = 1e-8) -> torch.Tensor:
    return Sigma + torch.eye(Sigma.shape[0], device=Sigma.device) * damp



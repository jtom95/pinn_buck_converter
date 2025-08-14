from typing import Callable, Tuple, Optional, Union
import torch
from abc import ABC, abstractmethod

from ..config import Parameters
from .residuals import ResidualFunc

class LikelihoodLossFunction(ABC):
    @abstractmethod
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sum_result: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the loss between predictions and targets.
        This method should be implemented by subclasses.
        """
        pass


class PriorLossFunction(ABC):
    @abstractmethod
    def __call__(self, logparams: Parameters, mu0: Parameters, sigma0: Parameters, **kwargs) -> torch.Tensor:
        """Compute the prior loss for the given parameters.
        This method should be implemented by subclasses.
        """
        pass

class LossFunctionFactory:
    @staticmethod
    def wrap_likelihood(likelihood_func: Callable) -> LikelihoodLossFunction:
        """
        Dynamically wrap a normal function into a LikelihoodLossFunction implementation.
        """

        class LikelihoodLossFunctionImpl(LikelihoodLossFunction):
            def __call__(
                self,
                pred: torch.Tensor,
                target: torch.Tensor,
                sum_result: bool = True,
                **kwargs,
            ) -> torch.Tensor:
                return likelihood_func(
                    pred=pred,
                    target=target,
                    sum_result=sum_result,
                    **kwargs,
                )

        return LikelihoodLossFunctionImpl()

    @staticmethod
    def wrap_prior(prior_func: Callable) -> PriorLossFunction:
        """
        Dynamically wrap a normal function into a PriorLossFunction implementation.
        """

        class PriorLossFunctionImpl(PriorLossFunction):
            def __call__(self, logparams: torch.Tensor, mu0: Parameters, sigma0: Parameters, **kwargs) -> torch.Tensor:
                return prior_func(logparams, mu0=mu0, sigma0=sigma0, **kwargs)

        return PriorLossFunctionImpl()


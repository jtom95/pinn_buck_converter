from typing import Callable, Tuple, Optional, Union
import torch
from abc import ABC, abstractmethod

from ..config import Parameters

class LikelihoodLossFunction(ABC):
    @abstractmethod
    def __call__(
        self,
        fwd_pred: torch.Tensor,
        bck_pred: torch.Tensor,
        fwd_target: torch.Tensor,
        bck_target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the loss between predictions and targets.
        This method should be implemented by subclasses.
        """
        pass


class PriorLossFunction(ABC):
    @abstractmethod
    def __call__(self, logparams: Parameters, **kwargs) -> torch.Tensor:
        """Compute the prior loss for the given parameters.
        This method should be implemented by subclasses.
        """
        pass


class MAPLossFunction(ABC):
    @abstractmethod
    def __call__(
        self,
        parameter_guess: Parameters,
        preds: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the MAP loss for the given parameter guess, predictions, and targets.
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
                fwd_pred: torch.Tensor,
                bck_pred: torch.Tensor,
                fwd_target: torch.Tensor,
                bck_target: torch.Tensor,
                **kwargs,
            ) -> torch.Tensor:
                return likelihood_func(
                    fwd_pred=fwd_pred,
                    bck_pred=bck_pred,
                    fwd_target=fwd_target,
                    bck_target=bck_target,
                    **kwargs,
                )

        return LikelihoodLossFunctionImpl()

    @staticmethod
    def wrap_prior(prior_func: Callable) -> PriorLossFunction:
        """
        Dynamically wrap a normal function into a PriorLossFunction implementation.
        """

        class PriorLossFunctionImpl(PriorLossFunction):
            def __call__(self, logparams: torch.Tensor, **kwargs) -> torch.Tensor:
                return prior_func(logparams, **kwargs)

        return PriorLossFunctionImpl()

    @staticmethod
    def wrap_map_loss(map_loss_func: Callable) -> MAPLossFunction:
        """
        Dynamically wrap a normal function into a MAPLossFunction implementation.
        """

        class MAPLossFunctionImpl(MAPLossFunction):
            def __call__(
                self,
                parameter_guess: Parameters,
                preds: Tuple[torch.Tensor, torch.Tensor],
                targets: Tuple[torch.Tensor, torch.Tensor],
            ) -> torch.Tensor:
                return map_loss_func(parameter_guess, preds, targets)

        return MAPLossFunctionImpl()


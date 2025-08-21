from typing import Callable, Optional, Tuple, Union
import torch
from ..parameters.parameter_class import Parameters
from .loss_function_configs import LikelihoodLossFunction, PriorLossFunction, LossFunctionFactory
from .residuals import ResidualFunc, basic_residual
from .loss_function_archive import log_normal_prior


class MAPLoss:
    def __init__(
        self,
        initial_params: Parameters,
        initial_sigma: Parameters,
        loss_likelihood_function: LikelihoodLossFunction,
        residual_function: ResidualFunc = basic_residual,
        prior_function: Optional[PriorLossFunction] = log_normal_prior,
        prior_function_kwargs: Optional[dict] = None,
        weight_prior_loss: float = 1.0,
        weight_likelihood_loss: Union[float, torch.Tensor] = 1.0,
        **kwargs,
    ):
        if not isinstance(loss_likelihood_function, LikelihoodLossFunction):
            loss_likelihood_function = LossFunctionFactory.wrap_likelihood(loss_likelihood_function)
        if not isinstance(prior_function, PriorLossFunction):
            prior_function = LossFunctionFactory.wrap_prior(prior_function)

        self.initial_params = initial_params
        self.initial_sigma = initial_sigma
        self.loss_likelihood_function = loss_likelihood_function
        self.residual_function = residual_function
        self.prior_function = prior_function
        self.prior_function_kwargs = prior_function_kwargs or {} # Initialize to empty dict if None
        self.weight_prior_loss = weight_prior_loss
        self.weight_likelihood_loss = weight_likelihood_loss
        self.extra_kwargs = kwargs
        
    @property
    def weight_likelihood_loss(self) -> Union[float, torch.Tensor]:
        """Get the weight of the likelihood loss."""
        return self._weight_likelihood_loss
    @weight_likelihood_loss.setter
    def weight_likelihood_loss(self, value: Union[float, torch.Tensor]):
        """Set the weight of the likelihood loss."""
        if isinstance(value, torch.Tensor):
            if value.dim() == 1:
                value = value[..., None, None]
        self._weight_likelihood_loss = value
        
    def __call__(
        self,
        parameter_guess: Parameters,
        preds: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        ll = self.loss_likelihood_function(
            pred=preds,
            target=targets,
            residual_func=self.residual_function,
            sum_result=False,
            **self.extra_kwargs,
        )
        ll_weighted = (self.weight_likelihood_loss * ll).sum()


        prior = self.prior_function(
            parameter_guess, 
            mu0=self.initial_params, 
            sigma0=self.initial_sigma, 
            **self.prior_function_kwargs
            )
        prior_weighted = self.weight_prior_loss * prior

        return ll_weighted + prior_weighted

    def clone(
        self,
        initial_params: Optional[Parameters] = None,
        initial_sigma: Optional[Parameters] = None,
        loss_likelihood_function: Optional[LikelihoodLossFunction] = None,
        residual_function: Optional[ResidualFunc] = None,
        prior_function: Optional[PriorLossFunction] = None,
        prior_function_kwargs: Optional[dict] = None,
        weight_prior_loss: Optional[float] = None,
        weight_likelihood_loss: Optional[Union[float, torch.Tensor]] = None,
        **kwargs,
    ) -> "MAPLoss":
        initial_params = initial_params if initial_params is not None else self.initial_params
        initial_sigma = (
            initial_sigma if initial_sigma is not None else self.initial_sigma
        )
        loss_likelihood_function = (
            loss_likelihood_function
            if loss_likelihood_function is not None
            else self.loss_likelihood_function
        )
        residual_function = (
            residual_function if residual_function is not None else self.residual_function
        )
        prior_function = prior_function if prior_function is not None else self.prior_function
        weight_prior_loss = (
            weight_prior_loss if weight_prior_loss is not None else self.weight_prior_loss
        )
        weight_likelihood_loss = (
            weight_likelihood_loss
            if weight_likelihood_loss is not None
            else self.weight_likelihood_loss
        )

        # update the kwargs dictionaries
        prior_function_kwargs = {
            **(self.prior_function_kwargs or {}),
            **(prior_function_kwargs or {}),
        }
        extra_kwargs = {**(self.extra_kwargs or {}), **kwargs}

        return MAPLoss(
            initial_params=initial_params,
            initial_sigma=initial_sigma,
            loss_likelihood_function=loss_likelihood_function,
            residual_function=residual_function,
            prior_function=prior_function,
            prior_function_kwargs=prior_function_kwargs,
            weight_prior_loss=weight_prior_loss,
            weight_likelihood_loss=weight_likelihood_loss,
            **extra_kwargs,
        )

    @property
    def likelihood(self) -> "MAPLoss":
        return self.clone(weight_prior_loss=0.0)

    @property
    def prior(self) -> "MAPLoss":
        return self.clone(weight_likelihood_loss=0.0)


## Create MAP loss function for training
def build_map_loss(
    initial_params: Parameters,
    initial_sigma: Parameters,
    loss_likelihood_function: LikelihoodLossFunction,
    residual_function: ResidualFunc = basic_residual,
    prior_function: Optional[PriorLossFunction] = log_normal_prior,
    prior_function_kwargs: Optional[dict] = None,
    weight_prior_loss: float = 1.0,
    weight_likelihood_loss: Union[float, torch.Tensor] = 1.0,
    **kwargs,
):
    """
    Build a MAP loss function for the given initial parameters and uncertainty.
    This function returns a callable loss function that can be used in training.
    The loss function combines the log-normal prior and the forward-backward loss.
    """
    if not isinstance(loss_likelihood_function, LikelihoodLossFunction):
        # If the provided loss_likelihood_function is not a LikelihoodLossFunction, wrap it
        loss_likelihood_function = LossFunctionFactory.wrap_likelihood(loss_likelihood_function)

    if not isinstance(prior_function, PriorLossFunction):
        # If the provided prior_function is not a PriorLossFunction, wrap it
        prior_function = LossFunctionFactory.wrap_prior(prior_function)


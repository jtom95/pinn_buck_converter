from typing import Callable, Optional, Tuple, Union
import torch
from ..parameters.parameter_class import Parameters
from .loss_function_configs import LikelihoodLossFunction, PriorLossFunction, LossFunctionFactory
from ..parameter_transformation import make_log_param
from .residuals import ResidualFunc, basic_residual


## Likelihood Loss Functions
def loss_whitened(
    pred: torch.Tensor,
    target: torch.Tensor,
    residual_func: ResidualFunc,
    L: torch.Tensor,
    sum_result: bool = True,
) -> torch.Tensor:
    """
    Compute r^T Σ^{-1} r via Cholesky whitening: r -> z = L^{-1} r.
    """
    r = residual_func(pred, target)  # shape (B, T, 2)

    # take the last shape of r to get the number of state variables
    n_state_vars = r.shape[-1]

    # L has a Tx4x4 shape
    L_state_var = L[..., :n_state_vars, :n_state_vars]

    if L_state_var.dim() == 2:
        # if the L tensor has 2 dimensions, it is a single covariance matrix
        r_flat = r.view(-1, n_state_vars)  # flatten to (B*T, n_state_vars)
        z = torch.linalg.solve_triangular(L_state_var, r_flat.T, upper=False).T  # shape [N, n_state_vars]
        return 0.5 * z.square().sum()

    # if the L tensor has 3 dimensions, the first dimension is the number of transients
    if L_state_var.dim() != 3 or L_state_var.shape[1:] != (n_state_vars, n_state_vars):
        raise ValueError(f"L must have shape ({n_state_vars},{n_state_vars}) or (T,{n_state_vars},{n_state_vars})")

    # --- per‑transient covariances ------------------------------------------------
    # r : (B, T, 2)  → permute to (T, B, 2) so transient is leading batch dim
    r_permuted = r.permute(1, 0, 2)  # (T, B, 2)
    # RHS for solve_triangular must be (T, 2, B)
    z = torch.linalg.solve_triangular(L_state_var, r_permuted.transpose(1, 2), upper=False)  # (T, 2, B)

    ## IMPORTANT NOTE:
    # Note that using solve_triangular is much more stable for the gradients and optimization
    # compared to z = torch.matmul(r, L_inv.T)  # whitening: (B, 4)
    # even if L_inv = chol_inv(L) is used, it is still more stable to use solve_triangular
    # because it avoids the numerical issues with the inverse of the Cholesky factor.
    res_z = 0.5 * z.square()
    if sum_result:
        return res_z.sum()
    return res_z


def loss_whitened_fwbk(
    pred: torch.Tensor,
    target: torch.Tensor,
    residual_func: ResidualFunc,
    L_fwd: torch.Tensor,
    L_bck: torch.Tensor,
    sum_result: bool = True
) -> torch.Tensor:
    
    fwd_pred, bck_pred = pred
    fwd_target, bck_target = target
    
    loss_fwd = loss_whitened(
        fwd_pred,
        fwd_target,
        residual_func,
        L_fwd,
        sum_result=sum_result
    )

    loss_bck = loss_whitened(
        bck_pred,
        bck_target,
        residual_func,
        L_bck,
        sum_result=sum_result
    )

    return (loss_fwd + loss_bck) / 2


def loss_whitened_r_delta(
    fwd_pred: torch.Tensor,
    bck_pred: torch.Tensor,
    fwd_target: torch.Tensor,
    bck_target: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """
    Compute r^T Σ^{-1} r via Cholesky whitening: r -> z = L^{-1} r.
    """
    r = (fwd_pred - bck_pred) - (fwd_target - bck_target)

    if L.dim() == 2:
        # if the L tensor has 2 dimensions, it is a single covariance matrix
        r_flat = r.view(-1, 2)  # flatten to (B*T, 2)
        z = torch.linalg.solve_triangular(L, r_flat.T, upper=False).T  # shape [N, 2]
        return 0.5 * z.square().sum()

    # if the L tensor has 3 dimensions, the first dimension is the number of transients
    if L.dim() != 3 or L.shape[1:] != (2, 2):
        raise ValueError("L must have shape (2,2) or (T,2,2)")

    # --- per‑transient covariances ------------------------------------------------
    # r : (B, T, 2)  → permute to (T, B, 2) so transient is leading batch dim
    r_tb2 = r.permute(1, 0, 2)  # (T, B, 2)
    # RHS for solve_triangular must be (T, 2, B)
    z = torch.linalg.solve_triangular(L, r_tb2.transpose(1, 2), upper=False)  # (T, 2, B)

    ## IMPORTANT NOTE:
    # Note that using solve_triangular is much more stable for the gradients and optimization
    # compared to z = torch.matmul(r, L_inv.T)  # whitening: (B, 4)
    # even if L_inv = chol_inv(L) is used, it is still more stable to use solve_triangular
    # because it avoids the numerical issues with the inverse of the Cholesky factor.
    return 0.5 * z.square().sum()


## Prior Loss Functions
def log_normal_prior(logparams: Parameters, mu0: Parameters, sigma0: Parameters) -> torch.Tensor:
    """Return −log p(log z) assuming independent log-normal priors."""
    total = 0.0
    nominal_logparams = make_log_param(mu0)
    for name in logparams._frozen_keys:
        if name == "Rloads":
            # we have two lists of Rloads, so we need to iterate over them
            for i, rload in enumerate(logparams.Rloads):
                proposed_value = rload
                mu = nominal_logparams.Rloads[i]
                sig = sigma0.Rloads[i]
                total += ((proposed_value - mu) / sig) ** 2 / 2
        else:
            proposed_value = getattr(logparams, name)
            mu = getattr(nominal_logparams, name)
            sig = getattr(sigma0, name)
            total += ((proposed_value - mu) / sig) ** 2 / 2
    return total


class MAPLoss:
    def __init__(
        self,
        initial_params: Parameters,
        initial_uncertainty: Parameters,
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
        self.initial_uncertainty = initial_uncertainty
        self.loss_likelihood_function = loss_likelihood_function
        self.residual_function = residual_function
        self.prior_function = prior_function
        self.prior_function_kwargs = prior_function_kwargs
        self.weight_prior_loss = weight_prior_loss
        self.weight_likelihood_loss = weight_likelihood_loss
        self.extra_kwargs = kwargs

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

        if self.prior_function_kwargs is None:
            prior_kwargs = dict(
                nominal=self.initial_params,
                sigma=self.initial_uncertainty,
            )
        else:
            prior_kwargs = self.prior_function_kwargs

        prior = self.prior_function(parameter_guess, **prior_kwargs)
        prior_weighted = self.weight_prior_loss * prior

        return ll_weighted + prior_weighted

    def clone(self):
        return MAPLoss(
            initial_params=self.initial_params,
            initial_uncertainty=self.initial_uncertainty,
            loss_likelihood_function=self.loss_likelihood_function,
            residual_function=self.residual_function,
            prior_function=self.prior_function,
            prior_function_kwargs=self.prior_function_kwargs,
            weight_prior_loss=self.weight_prior_loss,
            weight_likelihood_loss=self.weight_likelihood_loss,
            **self.extra_kwargs,
        )

    def __repr__(self):
        return (f"MAPLoss(initial_params={self.initial_params}, "
                f"initial_uncertainty={self.initial_uncertainty}, "
                f"loss_likelihood_function={self.loss_likelihood_function}, "
                f"residual_function={self.residual_function}, "
                f"prior_function={self.prior_function}, "
                f"prior_function_kwargs={self.prior_function_kwargs}, "
                f"weight_prior_loss={self.weight_prior_loss}, "
                f"weight_likelihood_loss={self.weight_likelihood_loss}, "
                f"extra_kwargs={self.extra_kwargs})")

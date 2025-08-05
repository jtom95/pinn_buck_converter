from typing import Callable, Optional, Tuple, Union
import torch
from ..config import Parameters
from .loss_function_configs import LikelihoodLossFunction, PriorLossFunction, MAPLossFunction, LossFunctionFactory
from ..parameter_transformation import make_log_param



## Likelihood Loss Functions
def fw_bw_loss_whitened(
    fwd_pred: torch.Tensor,
    bck_pred: torch.Tensor,
    fwd_target: torch.Tensor,
    bck_target: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """
    Compute r^T Σ^{-1} r via Cholesky whitening: r -> z = L^{-1} r.
    """
    fwd_residual = (fwd_pred - fwd_target).view(-1, 2)  # shape (B*T, 2)
    bck_residual = (bck_pred - bck_target).view(-1, 2)  # shape (B*T, 2)

    r = torch.cat((fwd_residual, bck_residual), dim=1)  # (B, 4)
    z = torch.linalg.solve_triangular(L, r.T, upper=False).T  # shape [N, 4]
    ## IMPORTANT NOTE:
    # Note that using solve_triangular is much more stable for the gradients and optimization
    # compared to z = torch.matmul(r, L_inv.T)  # whitening: (B, 4)
    # even if L_inv = chol_inv(L) is used, it is still more stable to use solve_triangular
    # because it avoids the numerical issues with the inverse of the Cholesky factor.
    return 0.5 * (z**2).sum()


def blockwise_loss(
    fwd_pred: torch.Tensor,
    bck_pred: torch.Tensor,
    fwd_target: torch.Tensor,
    bck_target: torch.Tensor,
    Sigma: torch.Tensor,
):
    fwd_residual = (fwd_pred - fwd_target).view(-1, 2)  # shape (B*T, 2)
    bck_residual = (bck_pred - bck_target).view(-1, 2)  # shape (B*T, 2)

    Sig_fwfw = Sigma[:2, :2]
    Sig_bwbw = Sigma[2:, 2:]
    Sig_fwbw = Sigma[:2, 2:]

    L_fwfw = torch.linalg.cholesky(Sig_fwfw)
    L_bwbw = torch.linalg.cholesky(Sig_bwbw)

    z_fwfw = torch.linalg.solve_triangular(L_fwfw, fwd_residual.T, upper=False).T  # shape [N, 2]
    z_bwbw = torch.linalg.solve_triangular(L_bwbw, bck_residual.T, upper=False).T  # shape [N, 2]

    # Assuming Sigma_fwbw is usually not invertible
    lambda_cross = torch.mean(Sig_fwbw)  # heuristic; can be negative
    loss_cross = (fwd_residual * bck_residual).sum() * lambda_cross

    return 0.5 * (z_fwfw**2).sum() + 0.5 * (z_bwbw**2).sum() + loss_cross


def diag_second_order_loss(
    fwd_pred: torch.Tensor,
    bck_pred: torch.Tensor,
    fwd_target: torch.Tensor,
    bck_target: torch.Tensor,
    Sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Mahalanobis loss using diagonal precision blocks A and D
    with 2nd-order cross correction, but **no explicit B term**.
    No matrix inverses are formed – only Cholesky + triangular solves.
    """
    # residuals, flattened to (N,2)
    dtype = Sigma.dtype
    r_np1 = (fwd_pred - fwd_target).to(dtype).reshape(-1, 2)
    r_n = (bck_pred - bck_target).to(dtype).reshape(-1, 2)

    # split Σ into 2×2 blocks
    S_pp = Sigma[:2, :2]  # Σ_{++}
    S_mm = Sigma[2:, 2:]  # Σ_{--}
    S_pm = Sigma[:2, 2:]  # Σ_{+-}
    S_mp = S_pm.T  # Σ_{-+}

    eps = 1e-12
    eye = torch.eye(2, device=Sigma.device)

    # Cholesky factors (SPD guaranteed -> succeed)
    L_pp = torch.linalg.cholesky(S_pp + eps * eye)  # L_pp L_ppᵀ = Σ_{++}
    L_mm = torch.linalg.cholesky(S_mm + eps * eye)  # L_mm L_mmᵀ = Σ_{--}

    # Helper: apply Σ_{++}^{-1} to a 2-vector batch
    def apply_inv_pp(v):  # v shape (N,2)
        return torch.cholesky_solve(v.unsqueeze(-1), L_pp).squeeze(-1)

    # Helper: apply Σ_{--}^{-1}
    def apply_inv_mm(v):
        return torch.cholesky_solve(v.unsqueeze(-1), L_mm).squeeze(-1)

    # P r_{n+1}
    w = apply_inv_pp(r_np1)  # (N,2)

    # Build second-order correction for A term
    #   corr = Σ_{+-} Σ_{--}^{-1} Σ_{-+} (Σ_{++}^{-1} r)
    tmp = apply_inv_mm(w @ S_mp.T)  # (N,2)
    corrA = (S_pm @ tmp.T).T  # (N,2)

    # A-quadratic  rᵀ P r  +  rᵀ corr r
    loss_A = (w * r_np1).sum() + (w * corrA).sum()

    # Same for D block
    u = apply_inv_mm(r_n)  # (N,2)
    tmp2 = apply_inv_pp(u @ S_pm.T)
    corrD = (S_mp @ tmp2.T).T
    loss_D = (u * r_n).sum() + (u * corrD).sum()

    # Total loss (no cross-term B)
    return loss_A + loss_D


## Prior Loss Functions
def log_normal_prior(logparams: Parameters, nominal: Parameters, sigma: Parameters) -> torch.Tensor:
    """Return −log p(log z) assuming independent log-normal priors."""
    total = 0.0
    nominal_logparams = make_log_param(nominal)
    for name in Parameters._fields:
        if name == "Rloads":
            # we have two lists of Rloads, so we need to iterate over them
            for i, rload in enumerate(logparams.Rloads):
                proposed_value = rload
                mu = nominal_logparams.Rloads[i]
                sig = sigma.Rloads[i]
                total += ((proposed_value - mu) / sig) ** 2 / 2
        else:
            proposed_value = getattr(logparams, name)
            mu = getattr(nominal_logparams, name)
            sig = getattr(sigma, name)
            total += ((proposed_value - mu) / sig) ** 2 / 2
    return total



## Create MAP loss function for training
def build_map_loss(
    initial_params: Parameters,
    initial_uncertainty: Parameters,
    loss_likelihood_function: Union[LikelihoodLossFunction, Callable],
    prior_function: Optional[Union[PriorLossFunction, Callable]] = log_normal_prior,
    prior_function_kwargs: Optional[dict] = None,
    **kwargs,
) -> MAPLossFunction:
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

    def _map_loss(
        parameter_guess: Parameters,
        preds: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fwd_pred, bck_pred = preds
        fwd_target, bck_target = targets

        ll = loss_likelihood_function(
            fwd_pred=fwd_pred,
            bck_pred=bck_pred,
            fwd_target=fwd_target,
            bck_target=bck_target,
            **kwargs,
        )

        if prior_function_kwargs is None:
            # if no kwargs are provided, use the default arguments for the log_normal_prior function
            prior_kwargs = dict(
                nominal=initial_params,
                sigma=initial_uncertainty,
            )
        else:
            prior_kwargs = prior_function_kwargs

        prior = prior_function(parameter_guess, **prior_kwargs)
        map_loss = ll + prior
        return map_loss

    return LossFunctionFactory.wrap_map_loss(_map_loss)

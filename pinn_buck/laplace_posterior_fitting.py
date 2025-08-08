from dataclasses import dataclass
from typing import Callable, List, Literal

import torch
from torch.autograd.functional import hessian
from torch.func import functional_call
from scipy.stats import norm, lognorm
import numpy as np

from .model.loss_function_configs import MAPLossFunction
from .config import Parameters
from .model.model_param_estimator import BaseBuckEstimator

@dataclass
class LaplacePosterior:
    theta_log: torch.Tensor  # flat vector (num_params,)
    Sigma_log: torch.Tensor  # (num_params, num_params)
    theta_phys: torch.Tensor  # flat vector (num_params,)
    Sigma_phys: torch.Tensor  # (num_params, num_params)
    param_names: List[str]  # e.g. ["L","RL","C",...,"Rload1",...]

    # ----------------------------------------------------------------
    @property
    def gaussian_approx(self) -> List:
        """Univariate Normal for each *physical* parameter."""
        mean = self.theta_phys.cpu().numpy()
        var = np.diag(self.Sigma_phys.cpu().numpy())
        return [norm(loc=m, scale=np.sqrt(v)) for m, v in zip(mean, var)]

    # ----------------------------------------------------------------
    @property
    def lognormal_approx(self) -> List:
        """Univariate LogNormal for each parameter (because θ = exp(log θ))."""
        mu = self.theta_log.cpu().numpy()
        var = np.diag(self.Sigma_log.cpu().numpy())
        return [lognorm(s=np.sqrt(v), scale=np.exp(m)) for m, v in zip(mu, var)]

    # ----------------------------------------------------------------
    def print_param_uncertainty(
        self,
        distribution_type: Literal["gaussian", "lognormal"] = "lognormal",
        n_sigma: float = 1.0,
    ):
        """
        Print posterior mean and the symmetric ± n_sigma interval (Normal space)
        or its asymmetric counterpart (LogNormal).

        Parameters
        ----------
        distribution_type : 'gaussian' | 'lognormal'
            Whether to interpret the posterior as Gaussian (physical space)
            or LogNormal (parameters are exponentials of a Normal).
        n_sigma : float
            Width of the interval expressed in Normal-standard-deviation units
            (z-score). 1 → 68.27 %; 2 → 95.45 %; 3 → 99.73 %.
        """
        if distribution_type == "gaussian":
            mean = self.theta_phys.cpu().numpy()
            std = np.sqrt(np.diag(self.Sigma_phys.cpu().numpy()))
            for name, m, s in zip(self.param_names, mean, std):
                s = 0.0 if np.isnan(s) else s
                wid = n_sigma * s
                pct = 100.0 * wid / m if m != 0 else 0.0
                print(f"{name:10s}: {m:.3e}  ±{wid:.1e}  ({pct:.2f} %)")

        elif distribution_type == "lognormal":
            mu = self.theta_log.cpu().numpy()
            sg = np.sqrt(np.diag(self.Sigma_log.cpu().numpy()))
            for name, m, s in zip(self.param_names, mu, sg):
                s = 0.0 if np.isnan(s) else s
                mean_phys = np.exp(m + 0.5 * s**2)
                lower = np.exp(m - n_sigma * s)
                upper = np.exp(m + n_sigma * s)
                minus = mean_phys - lower
                plus = upper - mean_phys
                pct_minus = 100.0 * minus / mean_phys
                pct_plus = 100.0 * plus / mean_phys
                print(
                    f"{name:10s}: {mean_phys:.3e}  "
                    f"-{minus:.1e}/+{plus:.1e}  "
                    f"(-{pct_minus:.2f} %, +{pct_plus:.2f} %)"
                )
        else:
            raise ValueError("distribution_type must be 'gaussian' or 'lognormal'")


class LaplaceApproximator:
    """
    Laplace‐approximation helper for a trained **buck-converter estimator**.

    The class builds a *local* Gaussian approximation of the joint posterior
    over the **log–parameters** θ around the MAP estimate θ\_MAP that is already
    stored inside `model`.  The workflow is

       1.  **Flatten** every learnable log-parameter in a fixed, reproducible
           order (`_flat_vector_logparams`).
       2.  **Patch** the model with arbitrary θ via `torch.func.functional_call`
           so that the forward graph is fully differentiable wrt θ
           (`_build_loss_on_theta`).
       3.  Evaluate the user-supplied negative-log-posterior loss
           `loss_fn(θ, X)` at θ\_MAP, back–prop once, and call
           `torch.autograd.functional.hessian` to obtain ∇²\_θ ℓ.
       4.  Invert (damped) Hessian → posterior covariance **Σ\_log**.
       5.  Convert mean/covariance to *physical* parameter space via the
           diagonal Jacobian J = diag(exp θ\_MAP).

    The resulting `LaplacePosterior` exposes convenience routines to print
    1-σ / 2-σ intervals and to build independent Normal or Log-Normal
    marginals.
    """
    def __init__(
        self,
        model: BaseBuckEstimator,
        loss_fn: MAPLossFunction,
        device: torch.device = torch.device("cpu"),
        damping: float = 1e-6,
    ):
        """
        Parameters
        ----------
        model      : trained estimator that exposes
                     • `.named_parameters()`  containing  'log_*' tensors
                     • `._logparam_name_map()` → [('L','log_L'), ('Rload1','log_Rloads.0'), …]
                     • `.get_estimates()`      → Parameters in physical space
                     • `.logparams.iterator()` → iterator over *display* names
        loss_fn    : callable implementing  ℓ(θ, preds, targets)
                     (negative log-posterior, MAP objective)
        device     : device used for the heavy Hessian computation
        damping    : λ added to the diagonal of the Hessian to guarantee SPD
        """
        self.model = model
        self.loss_fn = loss_fn
        self.damping = damping
        self.device = device

        # collect every learnable log-parameter in a fixed order
        self._named, self._shapes = zip(
            *[(n, p.shape) for n, p in model.named_parameters() if n.startswith("log_")]
        )

        self._mapping = self.model._logparam_name_map()  # ordered list
        self._stored_names = [s for _, s in self._mapping]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _flat_vector_logparams(self) -> torch.Tensor:
        """
        Return θ_flat ∈ ℝᵈ with all scalar log-parameters concatenated
        **in the exact display order** given by `self._mapping`, which corresponds to the order of the parameters, of the iterator
        of the `Parameters` class, this is kept thanks to the _logparam_name_map() iterator in the model.

        The vector is detached from the graph and marked `requires_grad=True`
        so that PyTorch can form higher-order derivatives for the Hessian.
        """
        flat = [self.model.get_parameter(s).reshape(-1) for s in self._stored_names]
        return torch.cat(flat).detach().clone().requires_grad_()

    def _logparam_dictionary(self, theta_flat: torch.Tensor):
        """
        Split θ_flat back into a dictionary that can be fed to
        `torch.func.functional_call`.

        It is necessary for this function to take `theta_flat` as input, since it the
        `closure` function must be connected to `theta_flat`, even if `torch.func.functional_call`
        requires a dictionary of `self.model.named_parameters()`.

        Returns
        -------
        { stored_name : tensor(shape) }  where *stored_name* matches
        the keys in `self.model.named_parameters()`.
        """
        return {s: theta_flat[i] for i, s in enumerate(self._stored_names)}

    # Build loss function on θ.
    # This must be constructed so torch's autograd is calculated with respect to the parameters: this will allow to construct the Hessian for the Laplace approximation
    def _build_loss_on_theta(self, X: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Create a **closure** ℓ(θ_flat) that:

        1.  Maps θ_flat → {parameter-tensor}  (functional patch).
        2.  Builds a new `Parameters` object in *log* space so the parameter_guess of `loss_fn` is evaluated consistently.
        3.  Runs the patched model on `X` and computes loss against
            first-difference residual targets.

        The returned callable is differentiable wrt its θ argument – this is
        crucial for the Hessian-based Laplace approximation.
        """
        
        X = X.to(self.device)
        targets = (X[1:, :, :2].clone().detach(), X[:-1, :, :2].clone().detach())

        def closure(theta_vec: torch.Tensor) -> torch.Tensor:
            # 1) new param dict for functional_call
            new_param_dict = self._logparam_dictionary(theta_vec)
            kwargs_for_Parameters = {}
            Rloads = []
            for display, stored in self._mapping:
                log_param_value = new_param_dict[stored]
                if "Rload" in display:
                    Rloads.append(log_param_value)
                    continue
                kwargs_for_Parameters[display] = log_param_value
            kwargs_for_Parameters["Rloads"] = Rloads
            theta_logparams = Parameters(**kwargs_for_Parameters)
            # 3) predictions from the patched model
            preds = functional_call(self.model, new_param_dict, (X,))
            return self.loss_fn(parameter_guess=theta_logparams, preds=preds, targets=targets)

        return closure

    def fit(self, X: torch.Tensor) -> LaplacePosterior:
        """
        Run the Laplace procedure and return a `LaplacePosterior` object.

        Steps
        -----
        1.  θ_flat  ← current MAP weights from `model`
        2.  Build closure ℓ(θ_flat) and compute Hessian
        3.  Σ_log   = (H + λ I)^{-1}
        4.  Convert mean/cov via J = diag(exp θ_MAP)

        Returns
        -------
        LaplacePosterior  containing  
        • mean / covariance in log space and physical space  
        • list `param_names` matching the flat vector ordering
        """
        theta_map = self._flat_vector_logparams()
        loss_on_theta = self._build_loss_on_theta(X)

        # MAP loss at θ_map
        loss = loss_on_theta(theta_map)
        loss.backward()

        # Hessian & covariance in log-space
        H = hessian(loss_on_theta, theta_map)
        H = 0.5 * (H + H.T)  # symmetrise
        I = torch.eye(H.shape[0], device=H.device)
        Sigma_log = torch.linalg.inv(H + self.damping * I)

        # physical-space mean in same flat order
        est = self.model.get_estimates()
        theta_phys = torch.tensor([v for _, v in est.iterator()], device=Sigma_log.device)

        J = torch.diag(theta_phys)
        Sigma_phys = J @ Sigma_log @ J.T

        return LaplacePosterior(
            theta_log=theta_map.detach(),
            Sigma_log=Sigma_log,
            theta_phys=theta_phys,
            Sigma_phys=Sigma_phys,
            param_names=[d for d, _ in self._mapping],  # display order
        )

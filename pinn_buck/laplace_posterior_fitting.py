from dataclasses import dataclass
from typing import Callable, List, Literal, Union, Dict
import json
from pathlib import Path

import torch
from torch.autograd.functional import hessian
from torch.func import functional_call
from scipy.stats import norm, lognorm
from scipy.special import erf
import numpy as np

from .model.map_loss import MAPLoss
from .parameters.parameter_class import Parameters
from .constants import ParameterConstants
from .model.model_param_estimator import BaseBuckEstimator
from .parameters.parameter_class import Parameters

def _sigma_to_quantiles(n_sigma: float):
    """
    Convert a ±n_sigma interval into lower/upper quantiles.
    For example, n_sigma=1 → (0.1587, 0.8413) for a Normal.
    """
    alpha = 0.5 * erf(n_sigma / np.sqrt(2))
    return 0.5 - alpha, 0.5 + alpha

class LaplacePosteriorConstants:
    DEFAULT_FILE_PATTERN = "*.laplace.json"


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
        """
        Univariate LogNormal for each parameter in *physical units*,
        correcting for the model's internal scaling.

        Notes
        -----
        - The input `theta_log` and `Sigma_log` describe a Gaussian distribution
        in log-space for the *scaled* parameters log(param * scale). This method
        first subtracts `log(scale)` to recover the log-distribution of the true
        physical parameters.

        - The returned `scipy.stats.lognorm` objects use:
            s     = σ_log   (standard deviation in log-space)
            scale = exp(μ_log_phys)  (this is the median in physical space)

        - For a lognormal X ~ LogNormal(μ, σ²):
            * Median = exp(μ)  → 50% quantile on each side
            * Mean   = exp(μ + σ² / 2)  → larger than the median unless σ² = 0
            * Mode   = exp(μ - σ²)      → smaller than both mean and median

        The upward shift of the mean relative to the median arises from the
        moment-generating function (MGF) of the underlying Gaussian, evaluated
        at t=1:
            M_X(t) = exp(μ t + ½ σ² t²)  ⇒  E[X] = M_X(1) = exp(μ + ½ σ²)

        This skew explains why posterior summaries in lognormal space will
        generally differ from Gaussian summaries in physical space.
        """
        
        # 1) scaled log mean/cov from Laplace (these correspond to log(param * scale))
        mu_log = self.theta_log.cpu().numpy()
        var_log = np.diag(self.Sigma_log.cpu().numpy())

        # # 2) build name->scale map from the Parameters-based SCALE using the same iterator()
        # scale_map = {name: val for name, val in ParameterConstants.SCALE.iterator()}
        # # align scales to the LaplacePosterior ordering
        # scales = np.array([scale_map[name] for name in self.param_names], dtype=np.float64)

        # # 3) unscale in log-space:
        # # Currently the parameters are log(param * scale), in order to reobtain the original params:
        # # log(param) = log([param * scale]/scale) = log([param*scale]) - log(scale)
        # mu_phys = mu_scaled - np.log(scales)
        # var_phys = var_scaled  # unchanged when subtracting a constant

        # 4) scipy's lognorm: s = sigma, scale = exp(mu)
        return [lognorm(s=np.sqrt(v), scale=np.exp(m)) for m, v in zip(mu_log, var_log)]

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

        lower_q, upper_q = _sigma_to_quantiles(n_sigma)

        if distribution_type == "gaussian":
            dists = self.gaussian_approx
            for name, dist in zip(self.param_names, dists):
                mean = dist.mean()
                wid = mean - dist.ppf(lower_q)  # or just n_sigma * dist.std()
                pct = 100.0 * wid / mean if mean != 0 else 0.0
                print(f"{name:10s}: {mean:.3e}  ±{wid:.1e}  ({pct:.2f} %)")

        elif distribution_type == "lognormal":
            dists = self.lognormal_approx
            for name, dist in zip(self.param_names, dists):
                mean = dist.mean()
                lower = dist.ppf(lower_q)
                upper = dist.ppf(upper_q)
                minus = mean - lower
                plus = upper - mean
                pct_minus = 100.0 * minus / mean if mean != 0 else 0.0
                pct_plus = 100.0 * plus / mean if mean != 0 else 0.0
                print(
                    f"{name:10s}: {mean:.3e}  "
                    f"-{minus:.1e}/+{plus:.1e}  "
                    f"(-{pct_minus:.2f} %, +{pct_plus:.2f} %)"
                )

        else:
            raise ValueError("distribution_type must be 'gaussian' or 'lognormal'")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save LaplacePosterior to a JSON file.
        Tensors are stored as nested lists (CPU, float64).
        """
        path = Path(path)
        file_pattern = LaplacePosteriorConstants.DEFAULT_FILE_PATTERN
        
        # if it doesn't end in .laplace.json add it. If it just finishes in .json make it .laplace.json
        if not path.name.endswith(file_pattern):
            if path.name.endswith(".json"):
                path = path.with_name(path.stem + file_pattern[1:])
            else:
                path = path.with_suffix(file_pattern[1:])

        data = {
            "theta_log": self.theta_log.detach().cpu().numpy().tolist(),
            "Sigma_log": self.Sigma_log.detach().cpu().numpy().tolist(),
            "theta_phys": self.theta_phys.detach().cpu().numpy().tolist(),
            "Sigma_phys": self.Sigma_phys.detach().cpu().numpy().tolist(),
            "param_names": list(self.param_names),
        }
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(
        cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ) -> "LaplacePosterior":
        """
        Load LaplacePosterior from a JSON file.
        """        
        path = Path(path)
        if not path.name.endswith(LaplacePosteriorConstants.DEFAULT_FILE_PATTERN[1:]):
            raise ValueError(
                f"Invalid file pattern: {path.name}. Expected pattern: {LaplacePosteriorConstants.DEFAULT_FILE_PATTERN}"
            )
        with path.open("r") as f:
            data = json.load(f)

        return cls(
            theta_log=torch.tensor(data["theta_log"], dtype=torch.float32, device=device),
            Sigma_log=torch.tensor(data["Sigma_log"], dtype=torch.float32, device=device),
            theta_phys=torch.tensor(data["theta_phys"], dtype=torch.float32, device=device),
            Sigma_phys=torch.tensor(data["Sigma_phys"], dtype=torch.float32, device=device),
            param_names=data["param_names"],
        )


class LaplaceApproximator:
    """
    Laplace‐approximation helper for a trained **buck-converter estimator**.

    The class builds a *local* Gaussian approximation of the joint posterior
    over the **log–parameters** θ around the MAP estimate θ_MAP that is already
    stored inside `model`.  The workflow is

       1.  **Flatten** every learnable log-parameter in a fixed, reproducible
           order (`_flat_vector_logparams`).
       2.  **Patch** the model with arbitrary θ via `torch.func.functional_call`
           so that the forward graph is fully differentiable wrt θ
           (`_build_loss_on_theta`).
       3.  Evaluate the user-supplied negative-log-posterior loss
           `loss_fn(θ, X)` at θ_MAP, back–prop once, and call
           `torch.autograd.functional.hessian` to obtain ∇²_θ ℓ.
       4.  Invert (damped) Hessian → posterior covariance **Σ_log**.
       5.  Convert mean/covariance to *physical* parameter space via the
           diagonal Jacobian J = diag(exp θ_MAP).

    The resulting `LaplacePosterior` exposes convenience routines to print
    1-σ / 2-σ intervals and to build independent Normal or Log-Normal
    marginals.
    """
    def __init__(
        self,
        model: BaseBuckEstimator,
        loss_fn: MAPLoss,
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
        targets = self.model.targets(X).to(self.device)

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
    
    @property
    def param_names(self) -> List[str]:
        """List of parameter names in the order matching the flat vectors."""
        return [d for d, _ in self._mapping]

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

        # unscale the Sigma_log
        if self.model.param_rescaling is not None:
            scale_map = {name: val for name, val in self.model.param_rescaling.iterator()}
            # 2) build name->scale map from the Parameters-based SCALE using the same iterator()
            
            # align scales to the LaplacePosterior ordering
            scales = torch.tensor([scale_map[name] for name in self.param_names], dtype=torch.float64)

            # 3) unscale in log-space:
            # Currently the parameters are log(param * scale), in order to reobtain the original params:
            # log(param) = log([param * scale]/scale) = log([param*scale]) - log(scale)
            mu_log_phys = theta_map.detach() - torch.log(scales)
            # Variance unchanged when subtracting a constant
        else:
            mu_log_phys = theta_map.detach()
            # Variance unchanged when subtracting a constant

        # physical-space mean in same flat order
        est = self.model.get_estimates()
        theta_phys = torch.tensor([v for _, v in est.iterator()], device=Sigma_log.device)

        J = torch.diag(theta_phys)
        Sigma_phys = J @ Sigma_log @ J.T

        return LaplacePosterior(
            theta_log=mu_log_phys,
            Sigma_log=Sigma_log,
            theta_phys=theta_phys,
            Sigma_phys=Sigma_phys,
            param_names=self.param_names,  # display order
        )

    def fit_with_hac(
        self,
        X: torch.Tensor,
        L_r: torch.Tensor,
        *,
        residual_fn,
        max_lag: int | None = None,
        bartlett: bool = True,
        bandwidth_rule: str = "n13",
        center_scores: bool = True,
        weight_likelihood: float | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> LaplacePosterior:
        """
        NOT WORKING YET!
        Need to study the HAC covariance estimation theory better.
        """

        raise NotImplementedError(
            "HAC Laplace approximation is not implemented yet. "
            "Please use the standard `fit` method instead."
        )

        device = self.device
        X = X.to(device)

        # Likelihood weight used in your MAP objective (e.g., 1/VIF)
        if weight_likelihood is None:
            weight_likelihood = float(getattr(self.loss_fn, "weight_likelihood_loss", 1.0))

        # Bandwidth
        with torch.no_grad():
            targets = self.model.targets(X).to(device)
        N, T, C = targets.shape  # expect (N,T,2)
        if max_lag is None:
            M = (
                max(1, int(1.5 * (N ** (1 / 3))))
                if bandwidth_rule == "n13"
                else max(1, int(1.5 * (N ** (1 / 3))))
            )
            M = min(M, N - 1)
        else:
            M = max(1, min(max_lag, N - 1))
        w = torch.ones((M,), dtype=dtype, device=device)
        if bartlett:
            w = 1.0 - torch.arange(1, M + 1, device=device, dtype=dtype) / (M + 1)

        # θ_MAP and closure for full (posterior) loss
        theta_map = self._flat_vector_logparams().to(device)
        theta_map.requires_grad_(True)
        loss_on_theta = self._build_loss_on_theta(X)

        # Observed Hessian at MAP (full objective) + damping
        H = hessian(loss_on_theta, theta_map)
        H = 0.5 * (H + H.T)
        I = torch.eye(H.shape[0], device=device, dtype=H.dtype)
        H = (H + self.damping * I).to(dtype)

        # Helper to patch params
        def _param_dict(vec: torch.Tensor):
            return self._logparam_dictionary(vec)

        # Build preds function that depends on θ (NO .detach here!)
        def preds_with_theta(theta_vec: torch.Tensor) -> torch.Tensor:
            return functional_call(self.model, _param_dict(theta_vec), (X,))

        # Residuals at θ_MAP (for v = Σ^{-1} r); f_θ will be recomputed inside grads
        with torch.no_grad():
            preds_map = preds_with_theta(theta_map).to(dtype)
            targets = targets.to(dtype)
            r = residual_fn(preds_map, targets).to(dtype)  # (N,T,2)

        # L_r shape handling
        if L_r.ndim == 2:
            L = L_r.to(device=device, dtype=dtype).expand(T, -1, -1)  # (T,2,2)
        else:
            L = L_r.to(device=device, dtype=dtype)  # (T,2,2)

        # y = L^{-1} r; v = Σ^{-1} r = (L^{-T}) y
        rT = r.permute(1, 2, 0)  # (T,2,N)
        yT = torch.linalg.solve_triangular(L, rT, upper=False)  # (T,2,N)
        vT = torch.linalg.solve_triangular(L.transpose(-1, -2), yT, upper=True)
        v = vT.permute(2, 0, 1).contiguous()  # (N,T,2)

        # Scores s_{n,t} = ∂/∂θ [ weight * v_{n,t}^T f_θ(n,t) ]
        p = theta_map.numel()
        scores = torch.zeros((N, T, p), dtype=dtype, device=device)

        # We reuse computation graphs across (n,t); retain_graph to avoid re-building
        for t_idx in range(T):
            v_t = v[:, t_idx, :].detach()  # DETACH v (treated as constant)
            for n in range(N):

                def scalar_on_theta(theta_vec: torch.Tensor) -> torch.Tensor:
                    f_all = preds_with_theta(theta_vec).to(dtype)  # depends on θ
                    return weight_likelihood * (v_t[n] * f_all[n, t_idx]).sum()

                # gradient wrt θ (vector length p)
                grad_vec = torch.autograd.grad(
                    scalar_on_theta(theta_map),
                    theta_map,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                scores[n, t_idx] = grad_vec.to(dtype)

        if center_scores:
            scores = scores - scores.mean(dim=0, keepdim=True)

        # HAC G
        G = torch.zeros((p, p), dtype=dtype, device=device)
        Gamma0 = torch.einsum("ntp,ntq->pq", scores, scores) / N
        G = G + Gamma0
        for k in range(1, M + 1):
            Sk, S0 = scores[k:], scores[:-k]  # (N-k,T,p)
            Gamma_k = torch.einsum("ntp,ntq->pq", Sk, S0) / (N - k)
            wk = w[k - 1]
            G = G + wk * (Gamma_k + Gamma_k.T)

        # Sandwich: H^{-1} G H^{-1}
        H_inv = torch.linalg.inv(H)
        Sigma_log_robust = H_inv @ G @ H_inv

        # Physical-space transform
        est = self.model.get_estimates()
        theta_phys = torch.tensor([v for _, v in est.iterator()], device=device, dtype=dtype)
        J = torch.diag(theta_phys)
        Sigma_phys_robust = J @ Sigma_log_robust @ J.T

        post = LaplacePosterior(
            theta_log=theta_map.detach().to(dtype),
            Sigma_log=Sigma_log_robust,
            theta_phys=theta_phys,
            Sigma_phys=Sigma_phys_robust,
            param_names=[d for d, _ in self._mapping],
        )
        # (Optional) attach plain Laplace for reference
        post.Sigma_log_plain = H_inv
        post.Sigma_phys_plain = J @ H_inv @ J.T
        post.H = H
        post.G_hac = G
        post.hac_meta = {
            "M": int(M),
            "bartlett": bool(bartlett),
            "center_scores": bool(center_scores),
            "weight_likelihood": float(weight_likelihood),
        }
        return post

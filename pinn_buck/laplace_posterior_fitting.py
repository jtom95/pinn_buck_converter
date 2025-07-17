from dataclasses import dataclass
from typing import Callable, List, Literal

import torch
from torch.autograd.functional import hessian
from torch.func import functional_call
from scipy.stats import norm, lognorm
import numpy as np

from .config import Parameters


@dataclass
class LaplacePosterior:
    theta_log: torch.Tensor
    Sigma_log: torch.Tensor
    theta_phys: torch.Tensor
    Sigma_phys: torch.Tensor

    @property
    def gaussian_approx(self) -> List:
        """Returns a list of Gaussian distributions for the log-space parameters."""
        mean = self.theta_phys.cpu().numpy()
        cov = self.Sigma_phys.cpu().numpy()
        return [norm(loc=m, scale=np.sqrt(c)) for m, c in zip(mean, np.diag(cov))]
    @property
    def lognormal_approx(self) -> List:
        """Returns a list of LogNormal distributions for the physical-space parameters."""
        mu_log = self.theta_log.cpu().numpy()
        sigma_log = np.sqrt(np.diag(self.Sigma_log.cpu().numpy()))
        return [lognorm(s=s, scale=np.exp(m)) for m, s in zip(mu_log, sigma_log)]

    def print_param_uncertainty(
        self,
        distribution_type: Literal["gaussian", "lognormal"] = "lognormal",
        n_sigma: float = 1.0,  # 1 σ → 68 %, 2 σ → 95 %, 3 σ → 99.7 %
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
            for name, mu, sigma in zip(Parameters._fields, mean, std):
                sigma = 0.0 if np.isnan(sigma) else sigma
                mass_interval = n_sigma * sigma
                print(f"{name:8s}: {mu:.3e} ± {mass_interval:.1e}  ({100*mass_interval/mu:.2f} %)")

        elif distribution_type == "lognormal":
            mu_log = self.theta_log.cpu().numpy()
            sigma_log = np.sqrt(np.diag(self.Sigma_log.cpu().numpy()))
            for name, m, s in zip(Parameters._fields, mu_log, sigma_log):
                s = 0.0 if np.isnan(s) else s

                # Posterior mean (physical space)
                mean_phys = np.exp(m + 0.5 * s**2)

                # Lower / upper bounds of the central interval in physical space
                lower = np.exp(m - n_sigma * s)
                upper = np.exp(m + n_sigma * s)

                minus = mean_phys - lower  # negative side width
                plus = upper - mean_phys  # positive side width

                print(
                    f"{name:8s}: {mean_phys:.3e} "
                    f"-{minus:.1e}/+{plus:.1e}  "
                    f"(-{100*minus/mean_phys:.2f} %, +{100*plus/mean_phys:.2f} %)"
                )
        else:
            raise ValueError("distribution_type must be 'gaussian' or 'lognormal'")


class LaplaceApproximator:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        damping: float = 1e-6,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.damping = damping

    def get_map_logparams(self) -> torch.Tensor:
        """Flatten and prepare MAP estimate in log-space from trained model."""
        return (
            torch.cat([p.view(1) for p in self.model.logparams]).detach().clone().requires_grad_()
        )

    def _build_loss_fn_on_params(self, X: torch.Tensor, y: torch.Tensor) -> Callable:
        y_prev = X[:, :2].detach().to(X.device)
        y = y.detach().to(X.device)

        def loss_on_params(theta_vec: torch.Tensor) -> torch.Tensor:
            param_keys = [f"log_{n}" for n in Parameters._fields]
            split_vals = [v.view_as(getattr(self.model, k)) for k, v in zip(param_keys, theta_vec)]

            new_params = {k: v for k, v in zip(param_keys, split_vals)}
            logparams = Parameters(*split_vals)

            preds = functional_call(self.model, new_params, (X, y))
            return self.loss_fn(logparams, preds, y_prev, y)

        return loss_on_params

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> LaplacePosterior:
        theta_map = self.get_map_logparams()
        loss_fn_theta = self._build_loss_fn_on_params(X, y)

        loss = loss_fn_theta(theta_map)
        loss.backward()

        H = hessian(loss_fn_theta, theta_map)
        H = (H + H.T) / 2  # Ensure symmetry
        Sigma_log = torch.linalg.inv(H + self.damping * torch.eye(H.shape[0], device=H.device))

        theta_map_phys = torch.tensor(
            [getattr(self.model.get_estimates(), name) for name in Parameters._fields],
            device=Sigma_log.device,
        )
        J = torch.diag(theta_map_phys)
        Sigma_phys = J @ Sigma_log @ J.T

        return LaplacePosterior(
            theta_log=theta_map.detach().clone(),
            Sigma_log=Sigma_log,
            theta_phys=theta_map_phys,
            Sigma_phys=Sigma_phys,
        )

    @staticmethod
    def _build_gaussian_approx(mean: np.ndarray, cov: np.ndarray):
        std = np.sqrt(np.diag(cov))
        return [norm(loc=m, scale=s) for m, s in zip(mean, std)]

    @staticmethod
    def _build_lognormal_approx(mu_log: np.ndarray, sigma_log: np.ndarray):
        return [lognorm(s=s, scale=np.exp(m)) for m, s in zip(mu_log, sigma_log)]

    @staticmethod
    def _print_uncertainty(theta_phys: torch.Tensor, Sigma_phys: torch.Tensor):
        std_phys = torch.sqrt(torch.diag(Sigma_phys))
        for i, name in enumerate(Parameters._fields):
            mean = theta_phys[i].item()
            std = std_phys[i].item()
            if np.isnan(std):
                std = 0.0
            print(f"{name:8s}: {mean:.3e} ± {std:.1e}  ({100*std/mean:.2f} %)")

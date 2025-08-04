# %%
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import h5py
import numpy as np


# change the working directory to the root of the project
sys.path.append(str(Path.cwd()))


from pinn_buck.config import Parameters
from pinn_buck.config import TRUE as TRUE_PARAMS, INITIAL_GUESS as INITIAL_GUESS_PARAMS
from pinn_buck.config import _SCALE

# load measurement interface
from pinn_buck.io import Measurement
from pinn_buck.noise import add_noise_to_Measurement

from pinn_buck.parameter_transformation import make_log_param, reverse_log_param
from pinn_buck.model.model_param_estimator import BuckParamEstimator
from pinn_buck.model.losses import l2_loss
from pinn_buck.io_model import TrainingRun

from pinn_buck.io import LoaderH5

# %%
from scipy.stats import lognorm
from pinn_buck.config import Parameters


# Nominals and linear-space relative tolerances
NOMINAL = Parameters(
    L=6.8e-4,
    RL=0.4,
    C=1.5e-4,
    RC=0.25,
    Rdson=0.25,
    Rloads= [3.3, 10.0, 6.8],  # Rload1, Rload2, Rload3
    Vin=46.0,
    VF=1.1,
)

REL_TOL = Parameters(
    L=0.50,
    RL=0.4,
    C=0.50,
    RC=0.50,
    Rdson=0.5,
    Rloads= [0.3, 0.3, 0.3],  # Rload1, Rload2, Rload3
    Vin=0.3,
    VF=0.3,
)


# %%
import torch
import torch.nn as nn


def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(123)
device = "cpu"

# %% [markdown]
# ## Block-Diagonalized Covariance vs Full Covariance Fitting

# %%
from typing import Callable, Union, Iterable
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from contextlib import contextmanager
import math

# let's define a function to convert relative tolerances to standard deviations
# using the log-normal distribution assumption.
# Previously, we assumed sigma = log(1 + rel_tol). This means we assume that the relative toleraces contain 1 standard deviation
# of the data. Although usually the relative tolerances are defined as 2 or 3 standard deviations, we will use 1 standard deviation
# since this is the worst case scenario.


def rel_tolerance_to_sigma(rel_tol: Parameters) -> Parameters:
    """Convert relative tolerances to standard deviations."""

    def _to_sigma(value: float) -> torch.Tensor:
        """Convert a relative tolerance to standard deviation."""
        return torch.log(torch.tensor(1 + value, dtype=torch.float32))

    return Parameters(
        L=_to_sigma(rel_tol.L),
        RL=_to_sigma(rel_tol.RL),
        C=_to_sigma(rel_tol.C),
        RC=_to_sigma(rel_tol.RC),
        Rdson=_to_sigma(rel_tol.Rdson),
        Rloads = [
            _to_sigma(rload) for rload in rel_tol.Rloads
        ],  # Rload1, Rload2, Rload3
        Vin=_to_sigma(rel_tol.Vin),
        VF=_to_sigma(rel_tol.VF),
    )


# define the log-normal prior for the parameters assuming independent priors distrubuted according to the log-normal distribution.
# See the formula above.
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


def ensure_positive_definite(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Ensure the Sigma matrix is positive definite by adding a small value to the diagonal.
    """
    return Sigma + torch.eye(Sigma.shape[0], device=Sigma.device) * eps


def noise_power_to_sigma(
    noise_power_i: float,
    noise_power_v: float,
) -> torch.Tensor:

    return torch.tensor(
        [[noise_power_i, 0.0], [0.0, noise_power_v]],
        dtype=torch.float32,
    )


def estimate_Sigma_r_2x2_blocks(sigma_x: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    Estimate the Sigma matrix for the 2x2 blocks.
    This function assumes that sigma_x is a 2x2 matrix and J is a 2x2 matrix.
    """
    if sigma_x.shape != (2, 2):
        raise ValueError("sigma_x must be a 2x2 matrix.")
    if J.shape != (2, 2):
        raise ValueError("J must be a 2x2 matrix.")

    J_inv = torch.linalg.inv(J)

    # Calculate the Sigma matrix for the 2x2 blocks
    Sigma_fwfw = J @ sigma_x @ J.T + sigma_x
    Sigma_fwbw = -J @ sigma_x - sigma_x @ J_inv.T
    Sigma_bwfw = -J_inv @ sigma_x - sigma_x @ J.T
    Sigma_bwbw = J_inv @ sigma_x @ J_inv.T + sigma_x

    return {
        "fwfw": Sigma_fwfw,
        "fwbw": Sigma_fwbw,
        "bwfw": Sigma_bwfw,
        "bwbw": Sigma_bwbw
    }


def estimate_sigma_fw_bw(
    sigma_x: torch.Tensor,
    J: torch.Tensor,
    calculate_diag_terms: bool = True,
):
    sigma_blocks = estimate_Sigma_r_2x2_blocks(sigma_x, J)

    # build the 4x4 Sigma matrix
    Sigma = torch.zeros((4, 4), device=sigma_x.device)
    Sigma[:2, :2] = sigma_blocks["fwfw"]  # top-left
    Sigma[2:, 2:] = sigma_blocks["bwbw"]  # bottom-right
    
    if calculate_diag_terms:
        Sigma[:2, 2:] = sigma_blocks["fwbw"]  # top-right
        Sigma[2:, :2] = sigma_blocks["bwfw"]  # bottom-left
    return Sigma


def data_noise_to_sigma(
    data_noise: Union[float, Iterable, torch.Tensor], 
    jac: torch.Tensor, 
    calculate_diag_terms: bool = True,
    damp: float = 1e-8
    ) -> torch.Tensor:
    """Parse data_noise and return the inverse covariance matrix Sigma_x_inv."""
    if isinstance(data_noise, float):
        a = data_noise
        b = data_noise
    elif isinstance(data_noise, torch.Tensor):
        if data_noise.shape != (2, 2):
            raise ValueError("If data_noise is a tensor, it must be 2x2.")
        a = data_noise[0, 0]
        b = data_noise[1, 1]
    elif isinstance(data_noise, Iterable):
        data_noise = list(data_noise)
        if len(data_noise) != 2:
            raise ValueError("If data_noise is iterable, it must be of length 2.")
        a = data_noise[0]
        b = data_noise[1]
    else:
        raise TypeError("data_noise must be float, 2-tensor, or iterable of length 2.")

    sigma_x = torch.diag(torch.tensor([a, b], dtype=torch.float32))
    sigma_r =  estimate_sigma_fw_bw(
        sigma_x=sigma_x,
        J=jac,
        calculate_diag_terms=calculate_diag_terms,
    )

    # ensure the Sigma matrix is positive definite
    sigma_r = ensure_positive_definite(sigma_r, eps=damp)
    return sigma_r

def chol_inv(mat: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """return (LLᵀ)⁻¹ᐟ² = L⁻ᵀ   where LLᵀ = mat (add jitter if needed)"""
    mat = mat + eps * torch.eye(mat.size(0), device=mat.device)
    L = torch.linalg.cholesky(mat)
    return torch.cholesky_inverse(L)  # same as L⁻ᵀ · L⁻¹

def chol(mat: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """return L where LLᵀ = mat (add jitter if needed)"""
    mat = mat + eps * torch.eye(mat.size(0), device=mat.device)
    L = torch.linalg.cholesky(mat)
    return L  # same as L⁻ᵀ · L⁻¹


def fw_bw_loss_whitened(
    pred_np1: torch.Tensor,
    pred_n: torch.Tensor,
    observations_np1: torch.Tensor,
    observations_n: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """
    Compute r^T Σ^{-1} r via Cholesky whitening: r -> z = L^{-1} r.
    """
    residual_np1 = pred_np1 - observations_np1  # shape (batch_size, 2)
    residual_n = pred_n - observations_n  # shape (batch_size, 2)
    
    # flatten the first two dimensions to get a 2D tensor
    residual_np1 = residual_np1.view(-1, 2)  # (B, 2)
    residual_n = residual_n.view(-1, 2)  # (B, 2)
    
    r = torch.cat((residual_np1, residual_n), dim=1)  # (B, 4)
    z = torch.linalg.solve_triangular(L, r.T, upper=False).T  # shape [N, 4]
    ## IMPORTANT NOTE:
    # Note that using solve_triangular is much more stable for the gradients and optimization
    # compared to z = torch.matmul(r, L_inv.T)  # whitening: (B, 4)
    # even if L_inv = chol_inv(L) is used, it is still more stable to use solve_triangular
    # because it avoids the numerical issues with the inverse of the Cholesky factor.
    return 0.5 * (z**2).sum()


# define the loss function for MAP estimation that combines the L2 loss and the log-normal prior.
def make_map_loss(
    nominal: Parameters, sigma0: Parameters, L: torch.Tensor, scale_factor: float = 1.0
) -> Callable:

    def _loss(logparams: Parameters, preds, targets):
        i_np, v_np, i_np1, v_np1 = preds
        y_n, y_np1 = targets
        pred_n = torch.stack((i_np, v_np), dim=-1)
        pred_np1 = torch.stack((i_np1, v_np1), dim=-1)
        ll = fw_bw_loss_whitened(pred_n=pred_n, pred_np1=pred_np1, observations_n=y_n, observations_np1=y_np1, L=L)
        prior = log_normal_prior(logparams, nominal, sigma0)
        map_loss = ll + prior
        return map_loss * scale_factor
        

    return _loss

# %%
from dataclasses import dataclass


# for simplicity let's define a dataclass for the training configurations
@dataclass
class AdamOptTrainingConfigs:
    savename: str = "saved_run"
    out_dir: Path = Path(".")
    lr: float = 1e-3
    epochs: int = 20_000
    device: str = "cpu"
    patience: int = 5000
    lr_reduction_factor: float = 0.5
    epochs_lbfgs: int = 1500
    lr_lbfgs: float = 1e-3
    history_size_lbfgs: int = 50
    max_iter_lbfgs: int = 10
    clip_gradient_adam: float = None
    save_every_adam: int = 1000
    save_every_lbfgs: int = 10


class NormalizerMeanStd:
    """A simple normalizer that normalizes the data using mean and standard deviation."""

    def __init__(self, x: torch.Tensor):
        """Initialize the normalizer with the mean and standard deviation of the data."""
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the data."""
        return (x - self.mean) / self.std
    def normalize_current(self, i: torch.Tensor) -> torch.Tensor:
        """Normalize the current data."""
        return (i - self.mean[:, 0]) / self.std[:, 0]
    def normalize_voltage(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize the voltage data."""
        return (v - self.mean[:, 1]) / self.std[:, 1]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the data."""
        return x * self.std + self.mean

    def denormalize_current(self, i: torch.Tensor) -> torch.Tensor:
        """Denormalize the current data."""
        return i * self.std[:, 0] + self.mean[:, 0]
    def denormalize_voltage(self, v: torch.Tensor) -> torch.Tensor:
        """Denormalize the voltage data."""
        return v * self.std[:, 1] + self.mean[:, 1]

    def normalize_model_predictions(
        self, preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the model predictions."""
        i_pn, v_pn, i_pnp1, v_pnp1 = preds
        return (
            self.normalize_current(i_pn),
            self.normalize_voltage(v_pn),
            self.normalize_current(i_pnp1),
            self.normalize_voltage(v_pnp1),
        )
    def denormalize_model_predictions(
        self, preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Denormalize the model predictions."""
        i_pn, v_pn, i_pnp1, v_pnp1 = preds
        return (
            self.denormalize_current(i_pn),
            self.denormalize_voltage(v_pn),
            self.denormalize_current(i_pnp1),
            self.denormalize_voltage(v_pnp1),
        )

    def normalize_y(
        self, y: torch.Tensor
    ) -> torch.Tensor:
        """Normalize the output data."""
        return (y - self.mean) / self.std


class BuckParamEstimator__(nn.Module):
    """Physics‑informed NN for parameter estimation in a buck converter."""

    def __init__(
        self,
        param_init: Parameters,
    ) -> None:
        super().__init__()
        self.initialize_log_parameters(param_init)

    def initialize_log_parameters(self, param_init: Parameters):
        log_params = make_log_param(param_init)
        
        self.log_L = nn.Parameter(log_params.L, requires_grad=True)
        self.log_RL = nn.Parameter(log_params.RL, requires_grad=True)
        self.log_C = nn.Parameter(log_params.C, requires_grad=True)
        self.log_RC = nn.Parameter(log_params.RC, requires_grad=True)
        self.log_Rdson = nn.Parameter(log_params.Rdson, requires_grad=True)
        self.log_Rloads = nn.ParameterList(
            [nn.Parameter(r, requires_grad=True) for r in log_params.Rloads]
        )
        self.log_Vin = nn.Parameter(log_params.Vin, requires_grad=True)
        self.log_VF = nn.Parameter(log_params.VF, requires_grad=True)

    # ----------------------------- helpers -----------------------------

    def _physical(self) -> Parameters:
        """Return current parameters in physical units (inverse scaling)."""
        return reverse_log_param(self.logparams)
    
    
    @property
    def logparams(self) -> Parameters:
        """Return current log‑space parameters."""
        return Parameters(
            L=self.log_L,
            RL=self.log_RL,
            C=self.log_C,
            RC=self.log_RC,
            Rdson=self.log_Rdson,
            Rloads = [rload for rload in self.log_Rloads],
            Vin=self.log_Vin,
            VF=self.log_VF,
        )

    def get_estimates(self) -> Parameters:
        """Return current parameters in physical units."""
        params = self._physical()
        return Parameters(
            L=params.L.item(),
            RL=params.RL.item(),
            C=params.C.item(),
            RC=params.RC.item(),
            Rdson=params.Rdson.item(),
            Rloads = [rload.item() for rload in params.Rloads],
            Vin=params.Vin.item(),
            VF=params.VF.item(),
        )

    # ---------------------- physics right‑hand sides -------------------
    @staticmethod
    def _di(i_k, v_k, S, p: Parameters):
        return -((S * p.Rdson + p.RL) * i_k + v_k - S * p.Vin + (1 - S) * p.VF) / p.L

    @staticmethod
    def _dv(i_k, v_k, S, p: Parameters, rload, di):
        return (p.C * p.RC * rload * di + rload * i_k - v_k) / (p.C * (p.RC + rload))

    # ------------------------------- forward --------------------------
    @staticmethod
    def _rk4_step(i, v, D, dt, p: Parameters, sign=+1):
        """
        Vectorized RK4 step for shape [..., 1].
        i, v, D, dt have shape [batch, n_transients, 1]
        p.Rloads is a list [Rload_0, Rload_1, ..., Rload_T-1]
        """
        dh = dt * sign

        # Build rload tensor of shape [1, 1] for broadcasting
        rload = torch.stack(p.Rloads).view(1, -1)

        def f(i_, v_):
            di = -((D * p.Rdson + p.RL) * i_ + v_ - D * p.Vin + (1 - D) * p.VF) / p.L
            dv = (p.C * p.RC * rload * di + rload * i_ - v_) / (p.C * (p.RC + rload))
            return di, dv

        k1_i, k1_v = f(i, v)
        k2_i, k2_v = f(i + 0.5 * dh * k1_i, v + 0.5 * dh * k1_v)
        k3_i, k3_v = f(i + 0.5 * dh * k2_i, v + 0.5 * dh * k2_v)
        k4_i, k4_v = f(i + dh * k3_i, v + dh * k3_v)

        i_new = i + dh / 6.0 * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        v_new = v + dh / 6.0 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return i_new, v_new
    
    def _rload_from_seq(self, N: int, run_lengths):
        """Rows are in transient order 0→1→2."""
        l1, l2, l3 = run_lengths
        k1 = min(N, l1)
        k2 = min(max(N - l1, 0), l2)
        k3 = max(N - l1 - l2, 0)

        parts = []
        
        # use the same device as the model parameters
        device = next(iter(self.parameters())).device
        
        p = self._physical()
        
        if k1:
            parts.append(torch.ones((k1, 1), device=device) * p.Rload1)  
        if k2:
            parts.append(torch.ones((k2, 1), device=device) * p.Rload2)
        if k3:
            parts.append(torch.ones((k3, 1), device=device) * p.Rload3)
        return torch.cat(parts, 0)
        
    
    def _rload_from_idx(self, run_idx: torch.LongTensor):
        """
        run_idx: [batch_size, n_transients]
        Each index points to which Rload to pick for each transient.
        Returns shape [batch_size, n_transients, 1]
        """
        p = self._physical()
        lookup = torch.stack([p.Rload1, p.Rload2, p.Rload3])  # shape [3]
        rload = lookup[run_idx]  # [batch_size, n_transients]
        return rload.unsqueeze(-1)
    
    def _rload_from_idx(self, run_idx: torch.LongTensor):
        """Rows may be arbitrarily shuffled; use run_idx to look up Rload."""
        p = self._physical()
        lookup = torch.stack([p.Rload1, p.Rload2, p.Rload3]).to(device=run_idx.device) 
        # note that it is necessary to use stack rather than create a tensor directly with tensor([p.Rload1, p.Rload2, p.Rload3])
        # because the torch.tensor() does not keep the autograd history, instead it copies the data and breaks the computational graph.
        return lookup[run_idx]


class BuckParamEstimator(BuckParamEstimator__):
    """Physics‑informed NN for parameter estimation in a buck converter."""

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: [batch, n_transients, 4] -> (i_n, v_n, D, dt)
        y: [batch, n_transients, 2] -> (i_np1, v_np1)
        """
        i_n, v_n = X[..., 0], X[..., 1]
        D, dt = X[..., 2], X[..., 3]

        i_np1, v_np1 = y[..., 0], y[..., 1]

        p = self._physical()

        # Forward and backward predictions (vectorized)
        i_np1_pred, v_np1_pred = self._rk4_step(i_n, v_n, D, dt, p, sign=+1)
        i_n_pred, v_n_pred = self._rk4_step(i_np1, v_np1, D, dt, p, sign=-1)

        return i_n_pred, v_n_pred, i_np1_pred, v_np1_pred


class Trainer:
    def __init__(
        self,
        model: BuckParamEstimator,
        loss_fn: Callable,
        optim_cfg: AdamOptTrainingConfigs,
        lbfgs_loss_fn: Callable = None,
        device="cpu",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.lbfgs_loss_fn = lbfgs_loss_fn if lbfgs_loss_fn is not None else loss_fn
        self.optim_cfg = optim_cfg
        self.device = device
        self.history = {"loss": [], "params": [], "lr": []}

    def fit(self, X, y, normalize_input: bool = True):
        X = X.detach().to(self.device)
        y = y.detach().to(self.device)
        y_prev = X[..., :2].clone().detach().to(self.device)

        X = X.to(self.device)
        y = y.to(self.device)
        y_prev = y_prev.to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.optim_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.optim_cfg.lr_reduction_factor,
            patience=self.optim_cfg.patience,
        )

        # Initialize the best loss
        best_loss = float("inf")

        if normalize_input:
            # normalize on the i and v of the input
            normalizer = NormalizerMeanStd(X[:, :2])        
            y_prev_norm = normalizer.normalize(y_prev)
            y_norm = normalizer.normalize(y)
        else:
            # no normalization
            y_prev_norm = y_prev
            y_norm = y

        for it in range(1, self.optim_cfg.epochs + 1):
            opt.zero_grad()
            preds = normalizer.normalize_model_predictions(self.model(X, y)) if normalize_input else self.model(X, y)

            loss: torch.Tensor = self.loss_fn(self.model.logparams, preds, (y_prev_norm, y_norm))
            loss.backward()

            if self.optim_cfg.clip_gradient_adam is not None:
                # Clip gradients to prevent exploding gradients
                old_gr = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_cfg.clip_gradient_adam)                                                      

            opt.step()
            scheduler.step(loss.item())

            if it % 1000 == 0:
                est = self.model.get_estimates()
                if loss.item() < best_loss:
                    best_loss = loss.item()

                # Collect gradients for scalar parameters
                scalar_param_names = ["L", "RL", "C", "RC", "Rdson", "Vin", "VF"]
                grads = [
                    getattr(self.model, f"log_{name}").grad.view(1)
                    for name in scalar_param_names
                    if getattr(self.model, f"log_{name}").grad is not None
                ]

                # Add gradients for Rloads
                for rload_param in self.model.log_Rloads:
                    if rload_param.grad is not None:
                        grads.append(rload_param.grad.view(1))

                # Compute gradient norm
                if grads:
                    gradient_vector = torch.cat(grads)
                    gradient_norm = gradient_vector.norm().item()
                else:
                    gradient_norm = float("nan")  # no gradients found (shouldn't happen during training)

                # Print parameter estimates
                print(
                    f"[Adam] Iteration {it}, gradient_norm {gradient_norm:4e}, loss {loss:4e}, Parameters:",
                    f"L={est.L:.3e}, RL={est.RL:.3e}, C={est.C:.3e}, ",
                    f"RC={est.RC:.3e}, Rdson={est.Rdson:.3e}, ",
                    f"Rloads=[{', '.join(f'{r:.3e}' for r in est.Rloads)}], ",
                    f"Vin={est.Vin:.3f}, VF={est.VF:.3e}",
                )

            if it % self.optim_cfg.save_every_adam == 0:
                est = self.model.get_estimates()
                # update the histories with the last Adam iteration
                self.history["loss"].append(loss.item())
                self.history["params"].append(est)
                self.history["lr"].append(opt.param_groups[0]["lr"])

        # # → LBFGS
        # LBFGS optimization tends to find stable solutions that also minimize the gradient norm.
        # This will be useful when we want to compute the Laplace posterior, which relies on the Hessian of the loss function.

        lbfgs_optim = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.optim_cfg.lr_lbfgs,
            max_iter=self.optim_cfg.max_iter_lbfgs,  # inner line-search iterations
            history_size=self.optim_cfg.history_size_lbfgs,  # critical for stability
        )

        nan_abort = True  # raise RuntimeError on NaN/Inf

        # ------------------------------------------------------------------
        #  Closure with finite checks
        # ------------------------------------------------------------------
        def closure():
            lbfgs_optim.zero_grad()

            pred = self.model(X, y)
            pred_norm = normalizer.normalize_model_predictions(pred) if normalize_input else pred
            loss_val = self.lbfgs_loss_fn(
                self.model.logparams, pred_norm, (y_prev_norm, y_norm)
            )

            # 1)  finite-loss check
            if not torch.isfinite(loss_val):
                message = "[LBFGS] Non-finite loss encountered"
                if nan_abort:
                    raise RuntimeError(message)
                else:
                    print(message)
                    return loss_val
            # 2)  clip gradients to prevent exploding gradients
            loss_val.backward()

            return loss_val

        # ------------------------------------------------------------------
        #  LBFGS training loop
        # ------------------------------------------------------------------
        for it in range(1, self.optim_cfg.epochs_lbfgs+1):
            try:
                loss = lbfgs_optim.step(closure)
            except RuntimeError as err:
                print(f"[LBFGS] Stopped at outer iter {it}: {err}")
                break

            # 3)  post-step parameter sanity
            with torch.no_grad():
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    print("[LBFGS] Non-finite parameter detected — aborting.")
                    break

            if it % 100 == 0:
                est = self.model.get_estimates()

                if loss.item() < best_loss:
                    best_loss = loss.item()

                # Collect gradients for scalar parameters
                scalar_param_names = ["L", "RL", "C", "RC", "Rdson", "Vin", "VF"]
                grads = [
                    getattr(self.model, f"log_{name}").grad.view(1)
                    for name in scalar_param_names
                    if getattr(self.model, f"log_{name}").grad is not None
                ]

                # Add gradients for Rloads
                for rload_param in self.model.log_Rloads:
                    if rload_param.grad is not None:
                        grads.append(rload_param.grad.view(1))

                # Compute gradient norm
                if grads:
                    gradient_vector = torch.cat(grads)
                    gradient_norm = gradient_vector.norm().item()
                else:
                    gradient_norm = float("nan")  # no gradients found (shouldn't happen during training)


                print(
                    f"[LBFGS] Iteration {self.optim_cfg.epochs + it}, gradient_norm {gradient_norm:4e}, loss {loss:4e},  Parameters:",
                    f"L={est.L:.3e}, RL={est.RL:.3e}, C={est.C:.3e}, "
                    f"RC={est.RC:.3e}, Rdson={est.Rdson:.3e}, "
                    f"Rloads=[{', '.join(f'{r:.3e}' for r in est.Rloads)}], "
                    f"Vin={est.Vin:.3f}, VF={est.VF:.3e}",
                )

            if it % self.optim_cfg.save_every_lbfgs == 0:
                est = self.model.get_estimates()
                # update the histories with the last BFGS iteration
                self.history["loss"].append(loss.item())
                self.history["params"].append(est)
                self.history["lr"].append(opt.param_groups[0]["lr"])

        # Save the history to a CSV file
        training_run = TrainingRun.from_histories(
            loss_history=self.history["loss"],
            param_history=self.history["params"],
        )

        # generate the output directory if it doesn't exist
        self.optim_cfg.out_dir.mkdir(parents=True, exist_ok=True)

        # if savename doesn't end with .csv, add it
        savename = self.optim_cfg.savename

        if not savename.endswith(".csv"):
            savename += ".csv"

        training_run.save_to_csv(self.optim_cfg.out_dir / savename)
        print("Concluded training.")

        print("Training completed successfully.")
        print(f"Best loss: {best_loss:.4e}")
        best_params = training_run.best_parameters
        opt_model = BuckParamEstimator(
            param_init = best_params,
        )

        self.training_run = training_run
        self.opt_model = opt_model

        return opt_model

# %%
## Noise Power
lsb_i = 10 / (2**12 - 1)  # 10 A full-scale current
lsb_v = 30 / (2**12 - 1)  # 30 V full-scale voltage

# i and v noise levels should probably be considered separately:

sigma_noise_ADC_i = 1 * lsb_i  # 1 LSB noise
sigma_noise_5_i = 5 * lsb_i  # 5 LSB noise
sigma_noise_10_i = 10 * lsb_i  # 10 LSB noise

sigma_noise_ADC_v = 1 * lsb_v  # 1 LSB noise
sigma_noise_5_v = 5 * lsb_v  # 5 LSB noise
sigma_noise_10_v = 10 * lsb_v  # 10 LSB noise

noise_power_ADC_i = sigma_noise_ADC_i**2
noise_power_5_i = sigma_noise_5_i**2
noise_power_10_i = sigma_noise_10_i**2

noise_power_ADC_v = sigma_noise_ADC_v**2
noise_power_5_v = sigma_noise_5_v**2
noise_power_10_v = sigma_noise_10_v**2


# LOAD THE PRECOMPUTED JACOBIAN
jacobian_dir = Path.cwd() / "RESULTS" / "Jacobains" / "N0"
J_av = torch.load(jacobian_dir / "jacobian.pt")


# Generate the Sigma matrix for the noise power
sigma_adc = data_noise_to_sigma(
    data_noise = (noise_power_ADC_i, noise_power_ADC_v),
    jac=J_av,
    calculate_diag_terms=False,
    damp = 1e-7
)

sigma_5 = data_noise_to_sigma(
    data_noise = (noise_power_5_i, noise_power_5_v),
    jac=J_av,
    calculate_diag_terms=False,
    damp = 1e-7
)

sigma_10 = data_noise_to_sigma(
    data_noise = (noise_power_10_i, noise_power_10_v),
    jac=J_av,
    calculate_diag_terms=False,
    damp = 1e-7
)


print(f"Sigma matrix for ADC noise:\n{sigma_adc}")
print(f"Sigma matrix for 5 LSB noise:\n{sigma_5}")
print(f"Sigma matrix for 10 LSB noise:\n{sigma_10}")


# L_inv_adc = chol_inv(sigma_adc)
# L_inv_5 = chol_inv(sigma_5)
# L_inv_10 = chol_inv(sigma_10)

L_adc = chol(sigma_adc)
L_5 = chol(sigma_5)
L_10 = chol(sigma_10)


# %%
damp = 1e-5

sigma_adc_full = data_noise_to_sigma(
    data_noise = (noise_power_ADC_i, noise_power_ADC_v),
    jac=J_av,
    calculate_diag_terms=True,
    damp = damp
)

sigma_5_full = data_noise_to_sigma(
    data_noise = (noise_power_5_i, noise_power_5_v),
    jac=J_av,
    calculate_diag_terms=True,
    damp = damp
)

sigma_10_full = data_noise_to_sigma(
    data_noise = (noise_power_10_i, noise_power_10_v),
    jac=J_av,
    calculate_diag_terms=True,
    damp = damp*10
)


print(f"Sigma matrix for ADC noise (full):\n{sigma_adc_full}")
print(f"Sigma matrix for 5 LSB noise (full):\n{sigma_5_full}")
print(f"Sigma matrix for 10 LSB noise (full):\n{sigma_10_full}")

L_inv_adc_full = chol_inv(sigma_adc_full)
L_inv_5_full = chol_inv(sigma_5_full)
L_inv_10_full = chol_inv(sigma_10_full)

L_adc_full = chol(sigma_adc_full)
L_5_full = chol(sigma_5_full)
L_10_full = chol(sigma_10_full)


frob_sigma_adc = torch.linalg.norm(sigma_adc_full, ord="fro")
frob_sigma_5 = torch.linalg.norm(sigma_5_full, ord="fro")
frob_sigma_10 = torch.linalg.norm(sigma_10_full, ord="fro")

det_sigma_adc = torch.linalg.det(sigma_adc_full)
det_sigma_5 = torch.linalg.det(sigma_5_full)
det_sigma_10 = torch.linalg.det(sigma_10_full)

print(f"Frobenius norm of Sigma for ADC noise: {frob_sigma_adc:.4e}")
print(f"Frobenius norm of Sigma for 5 LSB noise: {frob_sigma_5:.4e}")
print(f"Frobenius norm of Sigma for 10 LSB noise: {frob_sigma_10:.4e}")

print(f"Determinant of Sigma for ADC noise: {det_sigma_adc:.4e}")
print(f"Determinant of Sigma for 5 LSB noise: {det_sigma_5:.4e}")
print(f"Determinant of Sigma for 10 LSB noise: {det_sigma_10:.4e}")

# %%
def blockwise_loss(
    pred_np1: torch.Tensor,
    pred_n: torch.Tensor,
    observations_np1: torch.Tensor,
    observations_n: torch.Tensor,
    Sigma: torch.Tensor,
):
    residual_np1 = pred_np1 - observations_np1  # shape (batch_size, 2)
    residual_n = pred_n - observations_n  # shape (batch_size, 2)
    
    # flatten the first two dimensions to get a 2D tensor
    residual_np1 = residual_np1.view(-1, 2)  # (B*T, 2)
    residual_n = residual_n.view(-1, 2)  # (B*T, 2)

    Sig_fwfw = Sigma[:2, :2]
    Sig_bwbw = Sigma[2:, 2:]
    Sig_fwbw = Sigma[:2, 2:]

    L_fwfw = torch.linalg.cholesky(Sig_fwfw)
    L_bwbw = torch.linalg.cholesky(Sig_bwbw)

    z_fwfw = torch.linalg.solve_triangular(L_fwfw, residual_np1.T, upper=False).T  # shape [N, 2]
    z_bwbw = torch.linalg.solve_triangular(L_bwbw, residual_n.T, upper=False).T  # shape [N, 2]

    # Assuming Sigma_fwbw is usually not invertible
    lambda_cross = torch.mean(Sig_fwbw)  # heuristic; can be negative
    loss_cross = (residual_np1 * residual_n).sum() / lambda_cross

    return 0.5 * (z_fwfw**2).sum() + 0.5 * (z_bwbw**2).sum() + loss_cross


def make_map_loss_blockwise(
    nominal: Parameters, sigma0: Parameters, Sigma: torch.Tensor, scale_factor: float = 1.0
) -> Callable:
    """Create a loss function for MAP estimation with blockwise loss."""
    
    def _loss(logparams: Parameters, preds, targets):
        i_np, v_np, i_np1, v_np1 = preds
        y_n, y_np1 = targets
        pred_n = torch.stack((i_np, v_np), dim=-1)
        pred_np1 = torch.stack((i_np1, v_np1), dim=-1)
        ll = blockwise_loss(
            pred_np1=pred_np1,
            pred_n=pred_n,
            observations_np1=y_np1,
            observations_n=y_n,
            Sigma=Sigma,
        )
        prior = log_normal_prior(logparams, nominal, sigma0)
        map_loss = ll + prior
        return map_loss * scale_factor

    return _loss

from pinn_buck.model.model_param_estimator import measurement_to_tensors

set_seed(123)
device = "cpu"

# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"

out_dir = Path.cwd() / "RESULTS" / "Bayesian" / "FullSigma_MAP1"


run_configs = AdamOptTrainingConfigs(
    savename="adam_run.csv",
    out_dir=out_dir,
    lr=1e-3,
    epochs=15_000,
    device="cpu",
    patience=3000,
    lr_reduction_factor=0.5,
    epochs_lbfgs=1_000,
    lr_lbfgs=1e-3,
    history_size_lbfgs=100,
    max_iter_lbfgs=20,
    clip_gradient_adam=1e6,  
)


# load the transient data as unified numpy arrays
def load_data_to_model(meas: Measurement):
    """Load the data from a Measurement object and return the model."""
    # load the transient data as unified numpy arrays
    X, y = meas.data_stacked_transients

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    # Model
    return X_t, y_t


GROUP_NUMBER_DICT = {
    0: "ideal",
    1: "ADC_error",
    2: "Sync Error",
    3: "5 noise",
    4: "10 noise",
    5: "ADC-Sync-5noise",
    6: "ADC-Sync-10noise",
}

l_dict = {
    1: L_adc,  # ADC error
    3: L_5,  # 5 noise
    4: L_10,  # 10 noise
}
l_dict_full = {
    1: L_adc_full,  # ADC error
    3: L_5_full,  # 5 noise
    4: L_10_full,  # 10 noise
}

S_dict_full = {
    1: sigma_adc_full,  # ADC error
    3: sigma_5_full,  # 5 noise
    4: sigma_10_full,  # 10 noise
}


noisy_measurements = {}
trained_models = {}
trained_runs = {}
inverse = False

# Load the data from the hdf5 file
io = LoaderH5(db_dir, h5filename)

for idx, group_number in enumerate(l_dict.keys()):
    group_name = GROUP_NUMBER_DICT[group_number]
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    io.load(group_name)

    # Store the measurement in a dictionary
    noisy_measurements[group_name] = io.M

    run_configs.savename = f"noisy_run_{group_name}.csv"

    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")

    # Train the model on the noisy measurement
    X, y = load_data_to_model(
        meas=io.M,
    )
    model = BuckParamEstimator(param_init = NOMINAL).to(device)


    prior_info = {
        "nominal": NOMINAL,
        "sigma0": rel_tolerance_to_sigma(REL_TOL),
    }
    
    chol_L = l_dict[group_number]  # Cholesky factor of the noise covariance matrix
    chol_L_full = l_dict_full[group_number]  # Cholesky factor of the full noise covariance matrix
    
    sig_r = S_dict_full[group_number]  # full noise covariance matrix

    trainer = Trainer(
        model=model,
        loss_fn = make_map_loss(
            **prior_info,
            L=chol_L,  # Cholesky factor of the diagonal noise covariance matrix
            scale_factor=1.0,  # scale factor for the loss
        ),
        
        # loss_fn=make_map_loss(
        #     **prior_info,
        #     L = chol_L,  # Cholesky factor of the noise covariance matrix
        # ),
        optim_cfg=run_configs,
        device=device,
        lbfgs_loss_fn=make_map_loss_blockwise(
            **prior_info,
            Sigma=sig_r,  # use the full noise power for the group
            # scale_factor=np.sqrt(scale_factor_dict[group_number]),  # scale factor for the loss
        )
    )

    opt_model = trainer.fit(
        X=X,
        y=y,
        normalize_input=False
    )
    inverse = True  # inverse is False only for the ideal case, so we set it to True for the rest of the groups
    trained_models[group_name] = opt_model
    trained_runs[group_name] = trainer.training_run
    print("\n \n \n")

# %% [markdown]
# ## Laplace Approximation of the Posterior

# %%
from torch.autograd.functional import hessian
from torch.func import functional_call
from scipy.stats import norm, lognorm
import numpy as np
import torch
from torch import nn
from torch.autograd.functional import hessian
from torch.func import functional_call
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

from scipy.stats import norm, lognorm  # --- utilities

# -----------------------------------------------------------------
# user-supplied helpers
#   Parameters, make_log_param, reverse_log_param
#   log_normal_prior, rel_tolerance_to_sigma
#   likelihood_loss_triplets, _parse_data_noise_to_sigma
# must already be imported
# -----------------------------------------------------------------


# -----------------------------------------------------------------#
#   Container for the posterior                                   #
# -----------------------------------------------------------------#
@dataclass
class LaplacePosterior:
    theta_log: torch.Tensor  # MAP in log-space
    Sigma_log: torch.Tensor  # covariance in log-space
    theta_phys: torch.Tensor  # MAP in physical units
    Sigma_phys: torch.Tensor  # covariance in physical units


# -----------------------------------------------------------------#
#   LaplaceFitter class                                            #
# -----------------------------------------------------------------#
class LaplaceFitter:
    """
    Compute a Laplace (Gaussian) approximation to the posterior of a
    BuckParamEstimatorTriplets model.
    """

    # ------------- construction -----------------------------------
    def __init__(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
        damping: float = 1e-6,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.X = X.to(device)
        self.y = y.to(device)
        self.loss_fn = loss_fn
        self.damping = damping
        self.device = device

    # ------------- helper: flatten current log-parameters ---------
    def _flat_logparams(self) -> torch.Tensor:
        """Return  (P,)  vector, requires_grad=True."""
        vec = torch.cat([p.detach().clone().view(1) for p in self.model.logparams]).to(self.device)
        vec.requires_grad_(True)
        return vec

    # ------------- build closure L(θ) ------------------------------
    def _posterior_loss_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns f(theta_vec) that:
            1) rewrites model parameters,
            2) runs the triplet forward,
            3) computes −log posterior.
        """
        param_keys = [name for name, _ in self.model.named_parameters()]
        assert len(param_keys) == len(Parameters._fields), "param count mismatch"

        def loss(theta_vec: torch.Tensor) -> torch.Tensor:
            # split flat θ into individual tensors with correct shapes
            split = []
            offset = 0
            for name, p0 in self.model.named_parameters():
                n = p0.numel()
                split.append(theta_vec[offset : offset + n].view_as(p0))
                offset += n

            state_dict = {k: v for k, v in zip(param_keys, split)}  # new θ
            preds = functional_call(self.model, state_dict, (self.X, self.y))
            targets = (
                self.X[:, :2],  # previous time step (i, v)
                self.y,  # current time step (i, v)
            )
            return self.loss_fn(self.model, preds, targets)


        return loss

    # ------------- main entry -------------------------------------
    def fit(self) -> LaplacePosterior:
        theta_map = self._flat_logparams()
        loss_fn = self._posterior_loss_fn()

        # ----- compute MAP gradient once (optional sanity check) ---
        loss_map = loss_fn(theta_map)
        loss_map.backward()

        # ----- Hessian --------------------------------------------
        H = hessian(loss_fn, theta_map)
        H = (H + H.T) * 0.5  # symmetrise

        I = torch.eye(H.shape[0], device=self.device)
        Sigma_log = torch.linalg.inv(H + self.damping * I)

        # ----- convert to physical units ---------------------------
        theta_phys = torch.tensor(
            [getattr(self.model.get_estimates(), n) for n in Parameters._fields],
            device=self.device,
        )
        J = torch.diag(theta_phys)  # ∂θ_phys/∂θ_log = diag(θ_phys)
        Sigma_phys = J @ Sigma_log @ J.T

        return LaplacePosterior(
            theta_log=theta_map.detach(),
            Sigma_log=Sigma_log,
            theta_phys=theta_phys,
            Sigma_phys=Sigma_phys,
        )

    # -------- convenience static helpers --------------------------
    @staticmethod
    def build_gaussian_approx(mean: np.ndarray, cov: np.ndarray):
        std = np.sqrt(np.diag(cov))
        return [norm(loc=m, scale=s) for m, s in zip(mean, std)]

    @staticmethod
    def build_lognormal_approx(mu_log: np.ndarray, sigma_log: np.ndarray):
        return [lognorm(s=s, scale=np.exp(m)) for m, s in zip(mu_log, sigma_log)]

    @staticmethod
    def print_parameter_uncertainty(theta_phys, Sigma_phys):
        std_phys = torch.sqrt(torch.diag(Sigma_phys))
        for i, name in enumerate(Parameters._fields):
            mean = theta_phys[i].item()
            std = std_phys[i].item()
            pct = 100.0 * std / mean
            print(f"{name:8s}: {mean:.3e} ± {std:.1e} ({pct:.2f} %)")

# %%
damp = 1e-5

sigma_adc_full = data_noise_to_sigma(
    data_noise=(noise_power_ADC_i, noise_power_ADC_v),
    jac=J_av,
    calculate_diag_terms=True,
    damp=damp,
)

sigma_5_full = data_noise_to_sigma(
    data_noise=(noise_power_5_i, noise_power_5_v), jac=J_av, calculate_diag_terms=True, damp=damp
)

sigma_10_full = data_noise_to_sigma(
    data_noise=(noise_power_10_i, noise_power_10_v),
    jac=J_av,
    calculate_diag_terms=True,
    damp=damp,
)


print(f"Sigma matrix for ADC noise (full):\n{sigma_adc_full}")
print(f"Sigma matrix for 5 LSB noise (full):\n{sigma_5_full}")
print(f"Sigma matrix for 10 LSB noise (full):\n{sigma_10_full}")

L_inv_adc_full = chol_inv(sigma_adc_full)
L_inv_5_full = chol_inv(sigma_5_full)
L_inv_10_full = chol_inv(sigma_10_full)

noise_power_dict_full = {
    1: L_inv_adc_full,  # ADC error
    3: L_inv_5_full,  # 5 noise
    4: L_inv_10_full,  # 10 noise
}

noise_power_dict = {
    1: L_inv_adc,  # ADC error
    3: L_inv_5,  # 5 noise
    4: L_inv_10,  # 10 noise
}

# %%
from pinn_buck.laplace_posterior_fitting import LaplaceApproximator

lfits = {}
io = LoaderH5(db_dir, h5filename)

for number, power in noise_power_dict.items():
    label = GROUP_NUMBER_DICT[number]
    model = trained_models[label]

    lapl_approx = LaplaceApproximator(
        model=model,
        loss_fn=make_map_loss(
            nominal=NOMINAL,
            sigma0=rel_tolerance_to_sigma(REL_TOL),
            L_inv=power,  # use the noise power for the group
        ),
        damping=1e-6,
    )
    io.load(label)
    X, y = io.M.data

    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    lfit = lapl_approx.fit(X, y)
    lfits[label] = lfit

    print(f"\nParameter estimates for {label}:")
    lfit.print_param_uncertainty("gaussian")
    print("\n\n")

# %%
lfits = {}

for number, noise_power in noise_power_dict.items():
    label = GROUP_NUMBER_DICT[number]
    model = trained_models[label]
    
    print(f"Loading group {number}: {label}")
    io.load(label)


    # Train the model on the noisy measurement
    X, y, model = load_data_to_model(
        meas=io.M,
        initial_guess_params=NOMINAL,
    )

    # Fit Laplace posterior using the new class
    laplace = LaplaceFitter(
        model=model,
        X=X,
        y=y,
        loss_fn=make_map_loss(
            nominal=NOMINAL,
            sigma0=rel_tolerance_to_sigma(REL_TOL),
            L_inv=noise_power,  # use the full noise power for the group
        ),
        damping=1e-4,
        device="cpu",  # or "cuda" if using GPU
    )
    lfit = laplace.fit()

    # Compute Gaussian and LogNormal approximations
    gaussians = LaplaceFitter.build_gaussian_approx(
        mean=lfit.theta_phys.cpu().numpy(), cov=lfit.Sigma_phys.cpu().numpy()
    )

    lognormals = LaplaceFitter.build_lognormal_approx(
        mu_log=lfit.theta_log.cpu().numpy(),
        sigma_log=np.sqrt(torch.diag(lfit.Sigma_log).cpu().numpy()),
    )

    # Print and store
    print(f"\nParameter estimates for {label}:")
    LaplaceFitter.print_parameter_uncertainty(lfit.theta_phys, lfit.Sigma_phys)
    lfits[label] = lfit

# %%


# %%


# %%


# %%


# %%
tr.df.head()

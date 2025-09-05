# %%
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import h5py
import numpy as np


# change the working directory to the root of the project
project_root = Path.cwd()
sys.path.append(str(project_root))


from pinn_buck.parameters.parameter_class import Parameters
from pinn_buck.constants import ParameterConstants

# load measurement interface
from pinn_buck.io import Measurement
from pinn_buck.noise import add_noise_to_Measurement
from pinn_buck.data_noise_modeling.auxiliary import rel_tolerance_to_sigma

from pinn_buck.parameter_transformation import make_log_param, reverse_log_param
from pinn_buck.model.model_param_estimator import BuckParamEstimator, BaseBuckEstimator, BuckParamEstimatorFwdBck
from pinn_buck.model_results.history import TrainingHistory
from pinn_buck.model.loss_function_archive import loss_whitened, loss_whitened_fwbk

from pinn_buck.io import LoaderH5

# %%
from scipy.stats import lognorm
from pinn_buck.parameters.parameter_class import Parameters


# Nominals and linear-space relative tolerances
PRIOR_SIGMA = rel_tolerance_to_sigma(
    ParameterConstants.REL_TOL, number_of_stds_in_relative_tolerance=1
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


# def rel_tolerance_to_sigma(rel_tol: Parameters) -> Parameters:
#     """Convert relative tolerances to standard deviations."""

#     def _to_sigma(value: float) -> torch.Tensor:
#         """Convert a relative tolerance to standard deviation."""
#         return torch.log(torch.tensor(1 + value, dtype=torch.float32))

#     return Parameters(
#         L=_to_sigma(rel_tol.L),
#         RL=_to_sigma(rel_tol.RL),
#         C=_to_sigma(rel_tol.C),
#         RC=_to_sigma(rel_tol.RC),
#         Rdson=_to_sigma(rel_tol.Rdson),
#         Rloads = [
#             _to_sigma(rload) for rload in rel_tol.Rloads
#         ],  # Rload1, Rload2, Rload3
#         Vin=_to_sigma(rel_tol.Vin),
#         VF=_to_sigma(rel_tol.VF),
#     )


# %%
from dataclasses import dataclass

from pinn_buck.model.map_loss import MAPLoss
from pinn_buck.data_noise_modeling.jacobian_estimation import JacobianEstimator, JacobianEstimatorBase, FwdBckJacobianEstimator
from pinn_buck.data_noise_modeling.covariance_matrix_function_archive import covariance_matrix_on_basic_residuals, generate_residual_covariance_matrix, chol
from pinn_buck.model.trainer_auxiliary_functions import calculate_covariance_matrix, calculate_inflation_factor
from pinn_buck.model.residuals import basic_residual

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

noise_power_dict = {
    "ADC_error": (noise_power_ADC_i, noise_power_ADC_v),
    "5 noise": (noise_power_5_i, noise_power_5_v),
    "10 noise": (noise_power_10_i, noise_power_10_v),
}


# load measurements
db_dir = project_root.parent / "Databases"
h5filename = "buck_converter_Shuai_processed.h5"
io = LoaderH5(db_dir, h5filename)

### the jacobians are independent of the measurement so we can calculate them once
io.load("10 noise")
X = torch.tensor(io.M.data, device=device)
model = BuckParamEstimator(param_init = ParameterConstants.NOMINAL).to(device)


from typing import Optional
import itertools

class JacobianEstimator(JacobianEstimatorBase):
    # -------- single-time-index Jacobian on a (B,F) slice --------
    @staticmethod
    def estimate_J_single_t(
        Xi: torch.Tensor,  # (B, F) for one specific grid slice
        t: int,  # 0 .. B-2
        model: BaseBuckEstimator,  # estimator; must accept (B,1,F)
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Compute local Jacobian of one-step map at time t for a single (B,F) series.

        Returns:
            J : (S, F)  where S is the number of output state quantities
        """
        vec = Xi[t].clone().to(dtype).requires_grad_(True)

        def local_fwd(v):
            Xi_mod = Xi.clone()
            Xi_mod[t] = v
            fwd = model(Xi_mod.unsqueeze(1))  # expect (B-1, 1, S)
            # pick the output at the same time index t
            fwd_t = fwd[t].squeeze(0)  # (S,)
            return fwd_t

        J = torch.autograd.functional.jacobian(local_fwd, vec, create_graph=False)  # (S, F)
        return J

    # -------- full Jacobian across the parameter grid --------
    @classmethod
    def estimate_Jacobian(
        cls,
        X: torch.Tensor,  # shape (B, t1, t2, ..., F)
        model: BaseBuckEstimator,  # estimator with get_estimates()
        *,
        by_series: bool = True,  # kept for compatibility; irrelevant with multi-d grid
        number_of_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generalized Jacobian of the one-step RK4 map w.r.t. input X.

        The parameter-dependent transient grid comes from the *middle axes* of X:
            X.shape = (B, t1, t2, ..., F)

        For each grid index (i1, i2, ...), we:
            1) slice Xi = X[:, i1, i2, ..., :]  -> (B, F)
            2) build a fixed-parameter clone at that grid point
            3) average local Jacobians over sampled time indices

        Returns:
            out : tensor of shape (t1, t2, ..., F, S)
        """
        device = X.device
        B = X.shape[0]
        F = X.shape[-1]
        grid_shape = tuple(X.shape[1:-1])  # (t1, t2, ...)
        grid_nd = len(grid_shape)

        if B < 2:
            raise ValueError("Need at least 2 time rows to form one-step Jacobian (B>=2).")

        # --- choose time indices to sample
        max_idx = B - 2
        n_samples = (
            max_idx + 1 if number_of_samples is None else min(number_of_samples, max_idx + 1)
        )
        if n_samples == max_idx + 1:
            t_samples = torch.arange(n_samples, device=device)
        else:
            g = rng if rng is not None else torch.default_generator
            t_samples = torch.randperm(max_idx + 1, generator=g, device=device)[:n_samples]

        # --- determine output dimension S by a quick forward on one slice
        # pick first grid slice (or the only one)
        first_grid_idx = (0,) * grid_nd if grid_nd > 0 else ()
        Xi0 = X[(slice(None), *first_grid_idx, slice(None))]  # (B, F)
        # quick run
        with torch.no_grad():
            y0 = model(Xi0.unsqueeze(1))  # (B-1, 1, S)
        S = int(y0.shape[-1])

        # --- allocate output: (t1, t2, ..., F, S)
        out = torch.zeros(*grid_shape, F, S, dtype=dtype, device=device)

        # --- base params and their sequence layout
        base_params = model.get_estimates()  # Parameters (physical units)
        seq_keys, seq_lengths = cls._seq_keys_and_lengths(base_params)

        # --- loop over parameter grid
        grid_ranges: Iterable[range] = [range(n) for n in grid_shape] if grid_nd > 0 else [range(1)]
        for grid_idx in itertools.product(*grid_ranges):
            if grid_nd == 0:
                grid_idx = ()

            # 1) pick input slice
            Xi = X[(slice(None), *grid_idx, slice(None))].to(dtype).detach()  # (B, F)

            # 2) fix parameters at this grid point and clone the model
            fixed_params = cls._fix_params_at_index(base_params, grid_idx)
            model_clone = type(model)(param_init=fixed_params).to(device=device, dtype=dtype)
            model_clone.eval()

            # 3) accumulate Jacobians across sampled times
            Jsum = torch.zeros(S, F, dtype=dtype, device=device)
            for t in t_samples.tolist():
                Ji = cls.estimate_J_single_t(Xi, t, model_clone, dtype=dtype)  # (S, F)
                Jsum += Ji

            Jmean = (Jsum / float(n_samples)).transpose(0, 1)  # -> (F, S)
            out[grid_idx] = Jmean

        # rotate the grid so that the last two axes are (S, F): J = ∂y/∂x
        out = out.mT

        # Compatibility path (by_series flag kept for API parity):
        # with multi-d grids, by_series=False would mean averaging across all grid points.
        if not by_series and grid_nd > 0:
            # average over all grid axes -> (F, S)
            return out.mean(dim=tuple(range(0, grid_nd)))

        return out


jacs = {}
jacobian_estimator = JacobianEstimator()

for label, data_covariance in noise_power_dict.items():
    io.load(label)
    X = torch.tensor(io.M.data, device=device)
    jac = jacobian_estimator.estimate_Jacobian(
        X, model, 
        number_of_samples=500, 
        dtype=torch.float64
    )
    jacs[label] = jac

for label, jac in jacs.items():
    print(f"{label}:")
    for i in range(jac.shape[0]):
        print(f"Transient {i}, shape: {jac[i].shape}: {jac[i]}")

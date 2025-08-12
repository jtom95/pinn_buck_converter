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
from pinn_buck.model.model_param_estimator import BuckParamEstimator, BaseBuckEstimator
from pinn_buck.io_model import TrainingHistory
from pinn_buck.model.loss_function_archive import (
    build_map_loss,
    fw_bw_loss_whitened,
    diag_second_order_loss,
)

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


# %%
from dataclasses import dataclass

from pinn_buck.data_noise_modeling.auxiliary import rel_tolerance_to_sigma
from pinn_buck.data_noise_modeling.jacobian_estimation import estimate_Jacobian
from pinn_buck.data_noise_modeling.covariance_matrix_function_archive import covariance_matrix_on_standard_residuals
from pinn_buck.data_noise_modeling.covariance_matrix_config import (
    generate_residual_covariance_matrix,
    chol,
    chol_inv,
)


#%%


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

######################################


# load ideal noiseless measurement
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"
io = LoaderH5(db_dir, h5filename)
io.load("ideal")
X_ideal = io.M.torch_data


# add noise to the ideal measurement
def generate_noise(noise_power_i: float, noise_power_v: float, num_samples: int, num_transients: int=3, fill_with_zero_noise: bool=False, dtype=torch.float32) -> torch.Tensor:
    """
    Generate noise based on the noise power for current and voltage.
    The noise is generated as a 2D tensor with shape (B, T, 2), where T is the number of transients.
    """
    noise_i = torch.normal(mean=0, std=np.sqrt(noise_power_i), size=(num_samples, num_transients), dtype=dtype)
    noise_v = torch.normal(mean=0, std=np.sqrt(noise_power_v), size=(num_samples, num_transients), dtype=dtype)
    n_vars = 4 if fill_with_zero_noise else 2
    # fill the noise with zeros if specified
    noise = torch.zeros(num_samples, num_transients, n_vars, dtype=dtype)
    noise[:, :, 0] = noise_i
    noise[:, :, 1] = noise_v
    return noise


noise_adc = generate_noise(
    noise_power_ADC_i, noise_power_ADC_v, num_samples=X_ideal.shape[0], num_transients=X_ideal.shape[1], fill_with_zero_noise=True
)

noise_5 = generate_noise(
    noise_power_5_i, noise_power_5_v, num_samples=X_ideal.shape[0], num_transients=X_ideal.shape[1], fill_with_zero_noise=True
)

noise_10 = generate_noise(
    noise_power_10_i, noise_power_10_v, num_samples=X_ideal.shape[0], num_transients=X_ideal.shape[1], fill_with_zero_noise=True
)

# add noise to the ideal measurement data
X_adc = X_ideal + noise_adc 
X_5lsb = X_ideal + noise_5
X_10lsb = X_ideal + noise_10


### the jacobians are independent of the measurement so we can calculate them once
X = torch.tensor(io.M.data, device=device)
model = BuckParamEstimator(param_init = NOMINAL).to(device)

jacobians_stack_fwd = estimate_Jacobian(
    model, X, direction="forward", 
    by_series=True, 
    number_of_samples=500, 
    dtype=torch.float64
    )

jacobians_stack_bck = estimate_Jacobian(
    model, X, direction="backward",
    by_series=True,
    number_of_samples=500,
    dtype=torch.float64
    )
for i in range(jacobians_stack_fwd.shape[0]):
    print(f"Transient {i+1}:")
    print("Forward Jacobian:\n", jacobians_stack_fwd[i])
    print("Backward Jacobian:\n", jacobians_stack_bck[i])
    print("-" * 40)


# measurement_names = ["ADC_error", "5 noise", "10 noise"]
# jacobian_dict = {}
# for name in measurement_names:
#     print(f"Calculating Jacobian for {name} measurement...")
#     io = LoaderH5(db_dir, h5filename)
#     io.load(name)
#     # Train the model on the noisy measurement
#     jacobian_dict[name] = (jacobians_stack_fwd, jacobians_stack_bck)

# for name, (jacobians_stack_fwd, jacobians_stack_bck) in jacobian_dict.items():
#     print(f"Jacobian for {name}:")
#     for i in range(jacobians_stack_fwd.shape[0]):
#         print(f"Transient {i+1}:")
#         print("Forward Jacobian:\n", jacobians_stack_fwd[i])
#         print("Backward Jacobian:\n", jacobians_stack_bck[i])
#         print("-" * 40)

# crop the jacobians to be 2x2: only noise on i and v
jacobians_stack_fwd = jacobians_stack_fwd[:, :2, :2]
jacobians_stack_bck = jacobians_stack_bck[:, :2, :2]

def make_transient_covariance_matrices(
    data_covariance: Tuple[float, float],
    residual_covariance_block_func: Callable,
    jac_fwd: Iterable[torch.Tensor],
    jac_bck: Iterable[torch.Tensor],
    inlcude_diag_terms: bool = False,
    damp: float = 1e-7,
    dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    # check the number of fwd and bck jacobians is the same
    if len(jac_fwd) != len(jac_bck):
        raise ValueError("Number of forward and backward Jacobians must be the same.")
    covariance_matrices = []
    for jac_fwd_i, jac_bck_i in zip(jac_fwd, jac_bck):
        cov_matrix = generate_residual_covariance_matrix(
            data_covariance=data_covariance,
            residual_covariance_block_func=residual_covariance_block_func,
            jac_fwd=jac_fwd_i,
            jac_bck=jac_bck_i,
            include_diag_terms=inlcude_diag_terms,
            damp=damp,
            dtype=dtype
        )
        covariance_matrices.append(cov_matrix)
    return torch.stack(covariance_matrices)  # (T, 2, 2)


covariance_matrices_adc = make_transient_covariance_matrices(
    data_covariance=(noise_power_ADC_i, noise_power_ADC_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac_fwd=jacobians_stack_fwd,
    jac_bck=jacobians_stack_bck,
    inlcude_diag_terms=False,
    damp=1e-7,
    dtype=torch.float64
)

covariance_matrices_5 = make_transient_covariance_matrices(
    data_covariance=(noise_power_5_i, noise_power_5_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac_fwd=jacobians_stack_fwd,
    jac_bck=jacobians_stack_bck,
    inlcude_diag_terms=False,
    damp=1e-7,
    dtype=torch.float64
)

covariance_matrices_10 = make_transient_covariance_matrices(
    data_covariance=(noise_power_10_i, noise_power_10_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac_fwd=jacobians_stack_fwd,
    jac_bck=jacobians_stack_bck,
    inlcude_diag_terms=False,
    damp=1e-7,
    dtype=torch.float64
)


# print the covariance matrices for each transient

def print_transient_covariance_matrices(covariance_matrices: torch.Tensor):
    for i in range(covariance_matrices.shape[0]):
        print(f"Transient {i+1} Covariance Matrix:")
        print(covariance_matrices[i])

for name, matrices in zip(["ADC noise", "5 LSB noise", "10 LSB noise"], [covariance_matrices_adc, covariance_matrices_5, covariance_matrices_10]):
    print(f"Covariance matrices for {name}:")
    print_transient_covariance_matrices(matrices)
    print("-" * 40)

L_adc = chol(covariance_matrices_adc)
L_5 = chol(covariance_matrices_5)
L_10 = chol(covariance_matrices_10)


### Build full covariance matrices from the block builders


covariance_matrices_adc_full = make_transient_covariance_matrices(
    data_covariance=torch.tensor([noise_power_ADC_i, noise_power_ADC_v]),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac_fwd=jacobians_stack_fwd,
    jac_bck=jacobians_stack_bck,
    inlcude_diag_terms=True,
    damp=1e-7,
    dtype=torch.float64
)

covariance_matrices_5_full = make_transient_covariance_matrices(
    data_covariance=torch.tensor([noise_power_5_i, noise_power_5_v]),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac_fwd=jacobians_stack_fwd,
    jac_bck=jacobians_stack_bck,
    inlcude_diag_terms=True,
    damp=1e-7,
    dtype=torch.float64
)

covariance_matrices_10_full = make_transient_covariance_matrices(
    data_covariance=torch.tensor([noise_power_10_i, noise_power_10_v]),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac_fwd=jacobians_stack_fwd,
    jac_bck=jacobians_stack_bck,
    inlcude_diag_terms=True,
    damp=1e-7,
    dtype=torch.float64
)


# print the full covariance matrices
for name, matrices in zip(["ADC noise", "5 LSB noise", "10 LSB noise"], 
                            [covariance_matrices_adc_full, covariance_matrices_5_full, covariance_matrices_10_full]):
    print(f"Full Covariance Matrix for {name}:")
    print(matrices)
    print("-" * 40)

L_adc_full = chol(covariance_matrices_adc_full)
L_5_full = chol(covariance_matrices_5_full)
L_10_full = chol(covariance_matrices_10_full)


# %%
from pinn_buck.model.trainer import Trainer, TrainingConfigs


set_seed(123)
device = "cpu"
out_dir = Path.cwd() / "RESULTS" / "Bayesian" / "FullSigma_MAP1"


run_configs = TrainingConfigs(
    savename="adam_run.csv",
    out_dir=out_dir,
    lr_adam=1e-3,
    epochs_adam=10_000,
    device="cpu",
    patience=3000,
    lr_reduction_factor_adam=0.5,
    epochs_lbfgs=100,
    lr_lbfgs=1,
    history_size_lbfgs=20,
    max_iter_lbfgs=100,
    clip_gradient_adam=1e6, 
    save_every_lbfgs=1 
)


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
    1: covariance_matrices_adc_full,  # ADC error
    3: covariance_matrices_5_full,  # 5 noise
    4: covariance_matrices_10_full,  # 10 noise
}


noisy_measurements = {}
trained_models = {}
trained_runs = {}
inverse = False

# Load the data from the hdf5 file
# io = LoaderH5(db_dir, h5filename)

x_data = {
    "ADC noise": X_adc,
    "5 LSB noise": X_5lsb,
    "10 LSB noise": X_10lsb
}

for idx, group_number in enumerate(l_dict.keys()):
    group_name = GROUP_NUMBER_DICT[group_number]
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    io.load(group_name)

    # Store the measurement in a dictionary
    noisy_measurements[group_name] = io.M

for group_name, x in x_data.items():

    run_configs.savename = f"noisy_run_{group_name}.csv"

    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")

    # Train the model on the noisy measurement
    # X = torch.tensor(io.M.data, device=device)
    X = x.to(device)
    model = BuckParamEstimator(param_init = NOMINAL).to(device)

    chol_L = l_dict[group_number]  # Cholesky factor of the noise covariance matrix
    chol_L_full = l_dict_full[group_number]  # Cholesky factor of the full noise covariance matrix

    sig_r = S_dict_full[group_number]  # full noise covariance matrix

    trainer = Trainer(
        model=model,
        loss_fn=build_map_loss(
            initial_params=NOMINAL,
            initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
            loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
            L=chol_L,  # Cholesky factor of the diagonal noise covariance matrix
        ),
        cfg=run_configs,
        device=device,

        # lbfgs_loss_fn=build_map_loss(
        #     initial_params=NOMINAL,
        #     initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
        #     loss_likelihood_function=diag_second_order_loss,  # loss function for the forward-backward pass
        #     Sigma=sig_r,  # full noise covariance matrix
        # ),
        lbfgs_loss_fn=build_map_loss(
            initial_params=NOMINAL,
            initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
            loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
            L=chol_L,  # full noise covariance matrix
        ),
    )

    opt_model = trainer.fit(
        X=X
    )
    inverse = True  # inverse is False only for the ideal case, so we set it to True for the rest of the groups
    trained_models[group_name] = opt_model
    trained_runs[group_name] = trainer.history

    # test the loss function by evaluating the loss for the true parameters
    loss_diag_ideal = trainer.evaluate_loss(
        X=X,
        loss_fn=build_map_loss(
            initial_params=NOMINAL,
            initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
            loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
            L=chol_L,  # Cholesky factor of the diagonal noise covariance matrix
            weight_prior_loss=0.
        ),
        parameter_guess=TRUE_PARAMS
    )

    loss_full_ideal = trainer.evaluate_loss(
        X=X,
        loss_fn=build_map_loss(
            initial_params=NOMINAL,
            initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
            loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
            L=chol_L_full,  # full noise covariance matrix
            weight_prior_loss=0.
        ),
        parameter_guess=TRUE_PARAMS,
    )

    loss_full_best_Adams = trainer.evaluate_loss(
        X=X,
        loss_fn=build_map_loss(
            initial_params=NOMINAL,
            initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
            loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
            L=chol_L_full,  # full noise covariance matrix
            weight_prior_loss=0.
        ),
        parameter_guess=trainer.history.get_best_parameters('Adam')   
    )

    loss_best_BFGS = trainer.evaluate_loss(
        X=X,
        loss_fn=build_map_loss(
            initial_params=NOMINAL,
            initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
            loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
            L=chol_L_full,  # full noise covariance matrix
            weight_prior_loss=0.
        ),
        parameter_guess=trainer.history.get_best_parameters('LBFGS')   
    )

    print(f"True Param Loss for {group_name} (diagonal noise): {loss_diag_ideal:.6e} vs best found: {trainer.history.get_best_loss('Adam'):.6e}")
    print(f"True Param Loss for {group_name} (full noise): {loss_full_ideal:.6e} vs best found: {trainer.history.get_best_loss('LBFGS'):.6e}")

    TEMPLATE = "\t{idx}) {label:<40}{value:>12.6e}"
    print("FULL NOISE", group_name + ":")
    print(TEMPLATE.format(idx=1, label="Loss for TRUE PARAMETERS:", value=loss_full_ideal))
    print(TEMPLATE.format(idx=3, label="Loss for best with DIAGONAL COVARIANCE:",  value=loss_full_best_Adams))
    print(TEMPLATE.format(idx=2, label="Loss for best with FULL COVARIANCE:", value=loss_best_BFGS))

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

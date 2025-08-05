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

from pinn_buck.covariance_matrix_blocks_funcs import covariance_matrix_on_standard_residuals
from pinn_buck.covariance_matrix_auxil import generate_residual_covariance_matrix, chol, chol_inv

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
sigma_adc = generate_residual_covariance_matrix(
    data_covariance=(noise_power_ADC_i, noise_power_ADC_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac=J_av,
    include_diag_terms=False,
    damp=1e-7,
)

sigma_5 = generate_residual_covariance_matrix(
    data_covariance=(noise_power_5_i, noise_power_5_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac=J_av,
    include_diag_terms=False,
    damp=1e-7,
)

sigma_10 = generate_residual_covariance_matrix(
    data_covariance=(noise_power_10_i, noise_power_10_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac=J_av,
    include_diag_terms=False,
    damp=1e-7,
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

sigma_adc_full = generate_residual_covariance_matrix(
    data_covariance=(noise_power_ADC_i, noise_power_ADC_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac=J_av,
    include_diag_terms=True,
    damp=damp,
)

sigma_5_full = generate_residual_covariance_matrix(
    data_covariance=(noise_power_5_i, noise_power_5_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac=J_av,
    include_diag_terms=True,
    damp=damp,
)

sigma_10_full = generate_residual_covariance_matrix(
    data_covariance=(noise_power_10_i, noise_power_10_v),
    residual_covariance_block_func=covariance_matrix_on_standard_residuals,
    jac=J_av,
    include_diag_terms=True,
    damp=damp,
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
from pinn_buck.model.trainer import Trainer, TrainingConfigs


set_seed(123)
device = "cpu"

# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"

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
    X = torch.tensor(io.M.data, device=device)
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
            L=chol_L_full,  # full noise covariance matrix
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
        ),
        parameter_guess=trainer.history.get_best_parameters("LBFGS"),
    )

    print(f"True Param Loss for {group_name} (diagonal noise): {loss_diag_ideal:.6e} vs best found: {trainer.history.get_best_loss('Adam'):.6e}")
    print(f"True Param Loss for {group_name} (full noise): {loss_full_ideal:.6e} vs best found: {trainer.history.get_best_loss('LBFGS'):.6e}")

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

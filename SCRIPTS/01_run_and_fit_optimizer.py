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

from pinn_buck.data_noise_modeling.jacobian_estimation import estimate_Jacobian
from pinn_buck.data_noise_modeling.covariance_matrix_blocks_funcs import covariance_matrix_on_standard_residuals
from pinn_buck.data_noise_modeling.covariance_matrix_auxil import (
    generate_residual_covariance_matrix,
    chol,
    chol_inv,
)

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


# load measurements
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"
io = LoaderH5(db_dir, h5filename)

### the jacobians are independent of the measurement so we can calculate them once
io.load("10 noise")
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
from typing import Dict
from pinn_buck.model.trainer import Trainer, TrainingConfigs
from pinn_buck.laplace_posterior_fitting import LaplaceApproximator, LaplacePosterior

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
laplace_posteriors: Dict[str, LaplacePosterior] = {}
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
        # lbfgs_loss_fn=build_map_loss(
        #     initial_params=NOMINAL,
        #     initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
        #     loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
        #     L=chol_L_full,  # full noise covariance matrix
        # ),
    )

    trainer.fit(
        X=X
    )


    ### fit a Laplace Approximator for the posterior
    print("Fitting Laplace Posterior")
    laplace_posterior_approx = LaplaceApproximator(
        model=trainer.optimized_model(),
        loss_fn=build_map_loss(
            initial_params=NOMINAL,
            initial_uncertainty=rel_tolerance_to_sigma(REL_TOL),
            loss_likelihood_function=fw_bw_loss_whitened,  # loss function for the forward-backward pass
            L=chol_L,  # Cholesky factor of the diagonal noise covariance matrix
        ),
        device=device,
        damping=1e-7,
    )
    laplace_posterior = laplace_posterior_approx.fit(X)



    laplace_posteriors[group_name] = laplace_posterior
    trained_models[group_name] = trainer.optimized_model()
    trained_runs[group_name] = trainer.history


    print("Evaluating Test Losses")
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




for label, lfit in laplace_posteriors.items():
    print(f"\nParameter estimates for {label}:")
    lfit.print_param_uncertainty("gaussian")
    print("\n\n")

print("done")

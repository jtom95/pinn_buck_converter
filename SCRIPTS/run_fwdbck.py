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
model = BuckParamEstimatorFwdBck(param_init = ParameterConstants.NOMINAL).to(device)


covariance_matrices = []
jacobian_estimator = FwdBckJacobianEstimator()

for label, data_covariance in noise_power_dict.items():
    io.load(label)  
    X = torch.tensor(io.M.data, device=device)
    jac_fwd, jac_bck = jacobian_estimator.estimate_Jacobian(
        X, model, 
        number_of_samples=500, 
        dtype=torch.float64
    )[
        ..., :2, :2
    ]  # keep a size of (T, 2, 2)
    
    cov_matrix_fwd = generate_residual_covariance_matrix(
        data_covariance=data_covariance,
        residual_covariance_func=covariance_matrix_on_basic_residuals,
        jac=jac_fwd,
        dtype=torch.float64
    )
    
    cov_matrix_bck = generate_residual_covariance_matrix(
        data_covariance=data_covariance,
        residual_covariance_func=covariance_matrix_on_basic_residuals,
        jac=jac_bck,
        dtype=torch.float64
    )

    covariance_matrix = torch.stack([cov_matrix_fwd, cov_matrix_bck], dim=0)  # shape (2, T, 2, 2)
    covariance_matrices.append(covariance_matrix)


print("Covariance matrices and VIF factors calculated for ADC noise, 5 LSB noise, and 10 LSB noise.")
for idx in range(len(covariance_matrices)):
    # print(f"Group {idx+1} Covariance Matrix:")
    # print(covariance_matrices[idx])
    print(f"Group {idx+1}Variance Inflation Factor:")
    print(covariance_matrices[idx])
    print("-" * 40)

# print the covariance matrices for each transient

def print_transient_covariance_matrices(covariance_matrices: torch.Tensor):
    for i in range(covariance_matrices.shape[0]):
        print(f"Transient {i+1} Covariance Matrix:")
        print(covariance_matrices[i])

for name, matrices in zip(["ADC noise", "5 LSB noise", "10 LSB noise"], covariance_matrices):
    print(f"Covariance matrices for {name}:")
    print_transient_covariance_matrices(matrices)
    print("-" * 40)

l_dict = {key: dict(
    # fwd = chol(torch.eye(2)),
    # bck = chol(torch.eye(2)),
    fwd=chol(cov_matrix[0]),
    bck=chol(cov_matrix[1])
    )
        for key, cov_matrix in zip([1, 3, 4], covariance_matrices)
    }


# %%
from typing import Dict
from pinn_buck.model.trainer import Trainer, TrainingConfigs
from pinn_buck.laplace_posterior_fitting import LaplaceApproximator, LaplacePosterior

from pinn_buck.model.residuals import basic_residual
from pinn_buck.model.loss_function_archive import loss_whitened, loss_whitened_fwbk
from pinn_buck.model.map_loss import MAPLoss

set_seed(123)
device = "cpu"
out_dir = Path.cwd() / "RESULTS" / "LIKELIHOODS" / "FWD&BCK"
out_dir.mkdir(parents=True, exist_ok=True)

run_configs = TrainingConfigs(
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

noisy_measurements = {}
trained_models = {}
trained_runs = {}
laplace_posteriors: Dict[str, LaplacePosterior] = {}

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

    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")

    # Train the model on the noisy measurement
    X = torch.tensor(io.M.data, device=device)
    model = BuckParamEstimatorFwdBck(param_init = ParameterConstants.NOMINAL).to(device)

    L_fwd, L_bck = l_dict[group_number]["fwd"], l_dict[group_number]["bck"]

    map_loss = MAPLoss(
        initial_params=ParameterConstants.NOMINAL,
        initial_sigma=PRIOR_SIGMA,
        loss_likelihood_function=loss_whitened_fwbk,  # loss function for the forward-backward pass
        residual_function=basic_residual,
        L_fwd=L_fwd,  # Cholesky factor of the diagonal noise covariance matrix
        L_bck=L_bck,  # Cholesky factor of the diagonal noise covariance matrix
    ).likelihood

    trainer = Trainer(
        model=model,
        map_loss=map_loss,
        cfg=run_configs,
        device=device,
    )

    trainer.fit(
        X=X
    )

    ### fit a Laplace Approximator for the posterior
    print("Fitting Laplace Posterior")
    laplace_posterior_approx = LaplaceApproximator(
        model=trainer.optimized_model(),
        loss_fn=trainer.map_loss,
        device=device,
        damping=1e-7,
    )
    laplace_posterior = laplace_posterior_approx.fit(X)
    laplace_posterior.save(out_dir / f"laplace_posterior_{group_name}.json")

    laplace_posteriors[group_name] = laplace_posterior
    trained_models[group_name] = trainer.optimized_model()
    trained_runs[group_name] = trainer.history
    trainer.history.get_best_parameters().save(out_dir / f"best_params_{group_name}.json")
    trainer.history.save(out_dir / f"history_{group_name}")
    print("\n \n \n")


for label, lfit in laplace_posteriors.items():
    print(f"\nParameter estimates for {label}:")
    lfit.print_param_uncertainty("gaussian")
    print("\n\n")

print("done")

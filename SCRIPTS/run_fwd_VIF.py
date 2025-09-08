# %%
import os, sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict


# change the working directory to the root of the project
project_root = Path.cwd()
sys.path.append(str(project_root))


# import the necessary modules from the package
from circuit_parameter_estimator.data_loading_and_inspection.io import LoaderH5
from circuit_parameter_estimator.examples_archive.buck_converter import (
    BuckParamEstimator,
    ParameterArchive,
    BuckConverterParams,
    MeasurementGroupArchive,
)
from circuit_parameter_estimator.data_covariance.auxiliary import rel_tolerance_to_sigma
from circuit_parameter_estimator.optimization_loss.loss_function_archive import loss_whitened
from circuit_parameter_estimator.optimization_loss.map_loss import MAPLoss
from circuit_parameter_estimator.data_covariance.jacobian_estimation import JacobianEstimator
from circuit_parameter_estimator.data_covariance.covariance_matrix_function_archive import (
    covariance_matrix_on_basic_residuals,
    generate_residual_covariance_matrix,
    chol,
)
from circuit_parameter_estimator.residuals.residuals import basic_residual
from circuit_parameter_estimator.model_trainer.trainer import Trainer, TrainingConfigs
from circuit_parameter_estimator.laplace_posterior.fitting import (
    LaplaceApproximator,
    LaplacePosterior,
)


# %%  Nominals and linear-space relative tolerances
PRIOR_SIGMA = rel_tolerance_to_sigma(
    ParameterArchive.REL_TOL, number_of_stds_in_relative_tolerance=1
)
NOMINAL_VALUES = ParameterArchive.NOMINAL


# %% set random seeds for reproducibility
def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(123)
device = "cpu"

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

# %%
# Calculate the Cholesky factors and VIFs for each noise level on the fly, so it they can be updated dynamically during
# training


from typing import Tuple
from functools import partial
from circuit_parameter_estimator.residuals.residual_time_correlation import ResidualDiagnosticsGaussian
from circuit_parameter_estimator.model.model_base import BaseBuckEstimator

def calculate_Lr_and_vif(model: BaseBuckEstimator, X: torch.Tensor, data_covariance: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update the MAP loss function with the current model parameters."""
    jacobian_estimator = JacobianEstimator()
    jac = jacobian_estimator.estimate_Jacobian(
        X, model, number_of_samples=500, dtype=torch.float64
    )[..., :2, :2]
    covariance_matrix = generate_residual_covariance_matrix(
        data_covariance=data_covariance,
        residual_covariance_func=covariance_matrix_on_basic_residuals,
        jac=jac,
        dtype=torch.float64
    )

    Lr = chol(covariance_matrix)

    with torch.no_grad():
        pred = model(X)
        targets = model.targets(X)
        residuals = basic_residual(pred, targets)

    gauss_residual_diagnostics = ResidualDiagnosticsGaussian(residuals=residuals)
    vif = gauss_residual_diagnostics.quadloss_vif_from_residuals()
    vif_median = torch.median(vif)
    return Lr, vif_median


def callback_function_on_map(
    model: BaseBuckEstimator, 
    map_loss: MAPLoss,
    X: torch.Tensor, 
    data_covariance: Tuple[float, float]
    ) -> MAPLoss:
    
    """Update the MAP loss function with the current model parameters."""
    Lr, vif = calculate_Lr_and_vif(model, X, data_covariance)
    map_loss = map_loss.clone(
        L=Lr,  # Cholesky factor of the diagonal noise covariance matrix
        weight_likelihood_loss=1.0 / vif,  # use the VIF as the main loss
    )
    return map_loss


# print the VIFs for each noise level
Lrs = {}
vifs = {}
idx = 1

# load measurements
db_dir = project_root.parent / "Databases"
h5filename = "buck_converter_Shuai_processed.h5"
io = LoaderH5(db_dir, h5filename)
model = BuckParamEstimator(param_init = NOMINAL_VALUES).to(device)
for label, data_covariance in noise_power_dict.items():
    io.load(label)
    X = torch.tensor(io.M.data, device=device)
    with torch.no_grad():
        targets = model.targets(X)
        preds = model(X)
        residuals = basic_residual(preds, targets)
    # plot the residuals
    residual_diagnostic_gaussian = ResidualDiagnosticsGaussian(residuals=residuals)   
    vif_perchannel = residual_diagnostic_gaussian.per_channel_vif()
    vif_perchannel = torch.median(vif_perchannel, dim=0).values 

    Lr, vif = calculate_Lr_and_vif(model, X, data_covariance)
    Lrs[data_covariance] = Lr
    vifs[data_covariance] = vif

    print(f"GROUP {idx}) VIF: {vif} - per channel VIF: {vif_perchannel}")
    idx += 1


print("Covariance matrices and VIF factors calculated for ADC noise, 5 LSB noise, and 10 LSB noise.")


# Print the Cholesky factors for different noise levels
for idx, Lr in enumerate(Lrs.values()):
    # print(f"Group {idx+1} Covariance Matrix:")
    # print(covariance_matrices[idx])
    print(f"Group {idx+1} Cholesky factor:")
    print(Lr)
    print("-" * 40)

# %% # Train the model with different noise levels

# create output directory where the results will be saved
out_dir = Path.cwd() / "RESULTS" / "LIKELIHOODS" / "FWD_VIF"
out_dir.mkdir(parents=True, exist_ok=True)

# set the training configurations
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


# Dictionary mapping the different measurement trials numbers to their names
GROUP_NUMBER_DICT = MeasurementGroupArchive.SHUAI_ORIGINAL

noisy_measurements = {}
trained_models = {}
trained_runs = {}
laplace_posteriors: Dict[str, LaplacePosterior] = {}

# Load the data from the hdf5 file
io = LoaderH5(db_dir, h5filename)

for idx, group_name in enumerate(noise_power_dict.keys()):
    
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue

    print(f"Loading group: {group_name}")
    io.load(group_name)

    # Store the measurement in a dictionary
    noisy_measurements[group_name] = io.M

    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")

    # Train the model on the noisy measurement
    X = torch.tensor(io.M.data, device=device)
    model = BuckParamEstimator(param_init = NOMINAL_VALUES).to(device)

    data_covariance = noise_power_dict[group_name]

    map_loss = MAPLoss(
        initial_params=NOMINAL_VALUES,
        initial_sigma=PRIOR_SIGMA,
        loss_likelihood_function=loss_whitened,  # loss function for the forward-backward pass
        residual_function=basic_residual,
        # L=L,  # Cholesky factor of the diagonal noise covariance matrix
    ).likelihood

    callback_func = partial(
        callback_function_on_map,
        data_covariance=data_covariance
    )
    map_loss = callback_func(model, map_loss, X).clone(weight_likelihood_loss=1.)
    # # initially don't use vif which is too high for params far from the optimum
    # map_loss = map_loss_update_function(model, map_loss, X).clone(weight_likelihood_loss=1.0)
    
    trainer = Trainer(
        model=model,
        map_loss=map_loss,
        cfg=run_configs,
        device=device,
    )

    trainer.fit(
        X=X,
        update_loss_callback=callback_func,
        update_every_adam=3_000,
        update_every_lbfgs=30
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
    
    print("Final VIF:", 1 / trainer.map_loss.weight_likelihood_loss)
    print("\n \n \n")


for label, lfit in laplace_posteriors.items():
    print(f"\nParameter estimates for {label}:")
    lfit.print_param_uncertainty("gaussian")
    print("\n\n")

print("done")

# %% [markdown]
# ## Uncertainty Estimation
# 
# It was shown in the paper that it was possible to obtain reasonably good estimates of the parameters, even with noisy fields values. However, it is crucial for practical application to also have an estimation uncertainty. 
# 
# In order to obtain uncertainty estimation values, we need to introduce a Bayesian framework. This requires to introduce some prior distributions over the parameter values. We can immagine to have some nominal values for the components, or at least reasonable guesses or intervals for the values. 
# 
# **Proposed Bayesian Solution**:
# 
# **MAP estimate** & **Laplace estimate** around the posterior mode. With this approach the prior influences the optimization with via a penalty term on the loss. Gradient based optimization (Adam / LBFGS) can then be employed to find the optimum, which in this case would be the MAP estimate. Finally, we assume the posterior is a Gaussian and we calculate the best fit using the Laplace approximation, which relies on computing the Hessian.

# %% [markdown]
# ## Prior Distribution Family over the Physical Parameters

# %%
import os, sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict


# change the working directory to the root of the project
project_root = Path.cwd()
sys.path.append(str(project_root))

# %% [markdown]
# ### Prior distribution family:
# We have to consider **Additive vs. Multiplicative Variation**.
# 
# * **Additive variation** means the component deviates by **adding or subtracting** some noise:
# 
#   $$
#   x = x_0 + \epsilon
#   \quad\text{(e.g., } \epsilon \sim \mathcal{N}(0, \sigma^2)\text{)}
#   $$
# 
#   This is typical of **normal (Gaussian)** noise.
# 
# * **Multiplicative variation** means the component value varies by being **scaled** up or down:
# 
#   $$
#   x = x_0 \cdot (1 + \delta)
#   \quad\text{(e.g., } \delta \sim \mathcal{N}(0, \sigma^2)\text{)}
#   $$
# 
#   Or more generally:
# 
#   $$
#   \log x \sim \mathcal{N}(\mu, \sigma^2)
#   \Rightarrow x \sim \text{LogNormal}(\mu, \sigma^2)
#   $$
# 
#   This results in a **log-normal distribution** in linear space.
# 
# Considering how the components are manufactured:
# 
# * Component tolerances are often specified **as a percentage** (e.g., ±5%, ±10%).
# * This means that the error **scales** with the magnitude.
# * Example:
# 
#   * A 1 kΩ resistor with 5% tolerance → 950–1050 Ω
#   * A 10 kΩ resistor with 5% tolerance → 9500–10500 Ω
#     So the **absolute error grows** with the nominal value.
# 
# ---
# 
# ### Implication for Priors
# 
# If component values are specified with **percentage tolerances**, then we should model the distributions as:
# 
# $$
# \log(x) \sim \mathcal{N}(\log(x_0), \sigma^2)
# \quad\Rightarrow\quad
# x \sim \text{LogNormal}
# $$
# 
# This ensures:
# 
# * **Positivity**
# * **Correct scaling of uncertainty**
# * **Realistic tails** (e.g., 3σ errors reflect real-world max/min limits)
# 
# 
# A log-normal distribution is suitable because:
# 
# * All parameters are positive
# * Datasheet tolerances are multiplicative (e.g. ±20%)
# 
# For each parameter θ:
# 
# $$
# p(\theta) = \frac{1}{\theta \sigma \sqrt{2\pi}} \exp\left( -\frac{(\log \theta - \mu)^2}{2\sigma^2} \right)
# $$
# 
# Where:
# 
# * μ = log(nominal value)
# * σ = log(1 + relative tolerance)
# 
# ## Uniform Prior
# 
# If no prior knowledge is available, we can consider the prior to be a uniform distribution over a confidence interval. In this way, the contribution of the prior term is constant and does not influence the loss function minimization. 
# 
# This can be useful when we want to focus on finding the best likelihood functions for the data! 

# %%
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

import matplotlib.pyplot as plt

# %%  Nominals and linear-space relative tolerances
PRIOR_SIGMA = rel_tolerance_to_sigma(
    ParameterArchive.REL_TOL, number_of_stds_in_relative_tolerance=1
)
NOMINAL_VALUES = ParameterArchive.NOMINAL
TRUE_PARAMS = ParameterArchive.TRUE

# print the nominal parameters
print("Nominal Parameters:")
print(NOMINAL_VALUES)

print("Relative Tolerances:")
print(PRIOR_SIGMA)

# %%
from circuit_parameter_estimator.parameters.plotting_funcs import plot_parameter_priors

plot_parameter_priors(NOMINAL_VALUES, PRIOR_SIGMA, TRUE_PARAMS);
plt.show()

print("done")
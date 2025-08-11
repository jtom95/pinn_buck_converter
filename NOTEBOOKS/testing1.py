
# %%
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import h5py
import numpy as np


# change the working directory to the root of the project
sys.path.append(str(Path.cwd().parent))


from pinn_buck.config import Parameters

# load measurement interface
from pinn_buck.io import Measurement
from pinn_buck.noise import add_noise_to_Measurement

from pinn_buck.parameter_transformation import make_log_param, reverse_log_param
from pinn_buck.model.model_param_estimator import BuckParamEstimator
from pinn_buck.model.trainer import Trainer


from pinn_buck.io import LoaderH5

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

# %%
import matplotlib
from scipy.stats import lognorm
from pinn_buck.constants import ParameterConstants, MeasurementGroupArchive
from pinn_buck.data_noise_modeling.auxiliary import rel_tolerance_to_sigma

TRUE_PARAMS = ParameterConstants.TRUE
NOMINAL = ParameterConstants.NOMINAL
REL_TOL = ParameterConstants.REL_TOL
PRIOR_SIGMA = rel_tolerance_to_sigma(REL_TOL, number_of_stds_in_relative_tolerance=1) # transforms relative tolerance to the value of the standard deviation

# print the nominal parameters
print("Nominal Parameters:")
print(NOMINAL)

print("Relative Tolerances:")
print(REL_TOL)

# %%
from pinn_buck.model_results.history import TrainingHistory
from pinn_buck.model_results.ploting_comparisons import ResultsComparerTwo
# from pinn_buck.plot_aux import place_shared_legend


# Directories version (mirrors your originals)
fb_outdir = (
    Path.cwd() / "RESULTS" / "Testing" / "forward_vs_forward&backward" / "forward&backward"
)
f_outdir = Path.cwd() / "RESULTS" / "Testing" / "forward_vs_forward&backward" / "forward"

rc = ResultsComparerTwo.from_dirs(
    {
        "fwd": f_outdir,
        "fwd&bck": fb_outdir
    }
)

# Choose the labels you care about (ints map via the default dict; strings work too)
labels = (
    "ADC_error",
    "5 noise",
    "10 noise"
)

# Final % error comparison (side-by-side)
fig, ax = rc.plot_comparison(
    labels=labels, target=TRUE_PARAMS, select_lowest_loss=False
);

# Optional: tracked parameters for specific curves


fig, axes = rc.plot_tracked(target=TRUE_PARAMS, labels=labels);
plt.show()
print("done")
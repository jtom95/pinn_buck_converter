# %% [markdown]
# ## Uncertainty Estimation
#
# We have seen that it was possible to obtain reasonably good estimates of the parameters, even with noisy fields values. However, it is crucial for practical application to also have an estimation uncertainty.
#
# In order to obtain uncertainty estimation values, we need to introduce a Bayesian framework. This means introducing some priors over the parameters. This should not be an issue since we normally would have nominal values for the components, or at least reasonable guesses or intervals for the values.
#
# Next we need to decide how to frame the optimization in the Bayesian framework. Here are some options:
#
# 1. MAP estimate & Laplace estimate around the posterior mode: with this approach the prior influences the optimization with via a penalty term on the loss. Gradient based optimization (Adam) can then be employed to find the optimum, which in this case would be the MAP estimate. Finally, we assume the posterior is a Gaussian and we calculate the best fit using the Laplace approximation, which relies on computing the Hessian.
#
# 2. VI approach: should still be possible to use automatic-diff + Adam.
#
# 3. HMC / NUTS: Since we only have 10 dimensions, there is no curse of dimensionality. Should be possible to draw **exact values from the posterior** without relying on surrogate models.
#
# 4. Expectation Propagation (EP) Approaches: good with time-series. To be investigated.

# %% [markdown]
# ## MAP & Laplace

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

# %% [markdown]
# ### Parameter Nominal Values
#
# Now we need to define some nominal values for the parameters together with some realistic uncertainty estimates to simulate real engineering tasks.
#
# | Parameter | Description           | Nominal (N) | Tolerance / σ | Justification                      |
# | --------- | --------------------- | ----------- | ------------- | ---------------------------------- |
# | L         | Inductance            | 680 µH      | ±20%          | ±20% common for inductors          |
# | RL        | Inductor resistance   | 0.3 Ω       | ±0.1 Ω        | DCR variation or estimation        |
# | C         | Output capacitance    | 150 µF      | ±20%          | Electrolytics have wide tolerances |
# | RC        | ESR of output cap     | 0.25 Ω      | ±40%          | Datasheet often gives a max        |
# | Rdson     | Switch on-resistance  | 0.2 Ω       | ±10%          | MOSFETs vary with temperature      |
# | Rload1    | Load resistor 1       | 3.3 Ω       | ±5%           | Depends on load spec               |
# | Rload2    | Load resistor 2       | 10 Ω        | ±5%           | As above                           |
# | Rload3    | Load resistor 3       | 6.8 Ω       | ±5%           | As above                           |
# | Vin       | Input voltage         | 48 V        | ±2%           | From a regulated supply            |
# | VF        | Diode forward voltage | 0.9 V       | ±0.1 V        | Varies with current/temperature    |
#
#
# ---
#
# ### Prior distribution family:
#  we have to consider **Additive vs. Multiplicative Variation**.
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
# ## Notes
# it may be more realistic to set a normal or truncated normal for some parameters, e.g. V_in, R_loads, but for now let's assume a lognormal for all components.
#
# However, setting all priors to be a log-normal is a good initial choice, since:
#
# 1. Log-normal distributions naturally enforce positivity, which is a desirable property for **every parameter** in your model.
#
# 2. If the **relative standard deviation** is small (say < 10%), then a log-normal is **almost symmetric** and looks very much like a normal.
#
# 3. Modeling simplicity of using a log-normal for all parameters means:
#     * One consistent implementation for priors
#     * All prior PDFs live in log-space → simple KL terms in VI
#     * Posterior approximations (Laplace, variational) share the same structure
#
#     This pays off during training, debugging, and when visualizing uncertainty.
#
# ---
#
# ### ⚠️ Risks of using Log-Normal
#
# The core issue here is **how the prior interacts with the likelihood** to shape the posterior when the prior is **skewed** (like a log-normal) and the likelihood is **tight** (very confident).
#
# #### log-normal priors can become problematic in narrow posterior regimes
#
# Log-normal distributions are **asymmetric**:
#
# * The **mode** is less than the **mean**
# * The density decays **faster** on the left (toward 0) than on the right
#
# Now suppose the **likelihood (data)** is very confident about, say, $V_{in} = 48.1 \, \text{V}$, with very little uncertainty:
#
#   1. The prior is **skewed right**:
#
#       * Most mass is slightly above 48 V
#
#       * The mode is < 48 V (since log-normal mode = exp(μ − σ²))
#
#   2. The posterior, which is proportional to:
#
#   $$
#   \text{posterior} \propto \text{likelihood} \times \text{prior}
#   $$
#
#   gets **pulled** by this skewed prior.
#
#
#
# > If the data is highly informative and points to a value slightly **above** the nominal (e.g. 48.1 V), the **log-normal prior puts less mass there** than a normal would.
#     * The posterior mean gets **pulled lower**
#     * The posterior becomes **skewed left**
#     * The Laplace approximation (Gaussian) might **not match** the real shape
#
# So if your posterior is very concentrated and the prior is skewed, **even a small mismatch** between the likelihood peak and the prior mode can:
#
# * Shift the posterior
# * Lead to incorrect uncertainty quantification
# * Mislead downstream predictions if you're sampling
#     Result:
#
#
# ---
#
# ## ✅ Why a normal prior helps in this case
#
# A **normal prior** is symmetric. So:
#
# * It doesn't bias the posterior toward lower or higher values
# * The posterior stays centered where the data wants it to be
# * The Laplace (or mean-field VI) approximation is more accurate
#
# In summary, consider changing to a normal only if:
#
# * The posteriors for $V_{in}$ or $R_{\text{load}}$ are **very tightly concentrated**, and
# * Your Laplace or VI posterior is **asymmetric or biased** due to the log-normal prior's skew
#

# %%
import matplotlib
from scipy.stats import lognorm
from pinn_buck.config import Parameters


# Nominals and linear-space relative tolerances
NOMINAL = Parameters(
    L=6.8e-4,
    RL=0.4,
    C=1.5e-4,
    RC=0.25,
    Rdson=0.25,
    Rload1=3.3,
    Rload2=10.0,
    Rload3=6.8,
    Vin=46.0,
    VF=1.1,
)

REL_TOL = Parameters(
    L=0.50,
    RL=0.4,
    C=0.50,
    RC=0.50,
    Rdson=0.5,
    Rload1=0.3,
    Rload2=0.3,
    Rload3=0.3,
    Vin=0.3,
    VF=0.3,
)


# %% [markdown]
# ## Benchmark
#
# Let's inspect a normal run of the Adam optimizer with the nominal values as initial guess

# %%
import torch
import torch.nn as nn


def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_from_measurement_file(
    meas: Measurement,
    initial_guess_params: Parameters,
    savename: str = "saved_run",
    db_dir: Path = ".",
    lr: float = 1e-3,
    epochs: int = 30_000,
    device: str = "cpu",
    patience: int = 5000,
    lr_reduction_factor: float = 0.5,
):
    # load the transient data as unified numpy arrays
    X, y = meas.data
    s1, s2, s3 = list(
        map(lambda x: x - 1, meas.transient_lengths)
    )  # subtract 1 since we use the previous time step as input
    lb, ub = X.min(0), X.max(0)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)
    x0 = X_t[:, :2]

    # Model
    model = BuckParamEstimator(lb, ub, s1, s2, s3, initial_guess_params).to(device)

    history_loss = []
    history_params: List[Parameters] = []
    learning_rates: List[float] = []

    # --- tracking best loss ---
    best_loss, best_iter = float("inf"), -1

    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_reduction_factor, patience=patience
    )

    for it in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t, y_t)
        # loss = compute_loss(pred, x0, y_t) if it < epochs1 else compute_L1_loss(pred, x0, y_t)

        loss = l2_loss(pred, x0, y_t)
        loss.backward()

        optimizer.step()
        scheduler.step(loss)
        
        if it % 1000 == 0:
            est = model.get_estimates()
            history_loss.append(loss.item())
            history_params.append(est)
            # record the learning rate
            learning_rates.append(optimizer.param_groups[0]["lr"])

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_iter = it

            # print the parameter estimation
            est = model.get_estimates()
            print(
                f"Iteration {it}, loss {loss:4e},  Parameters (Adam):",
                f"L={est.L:.3e}, RL={est.RL:.3e}, C={est.C:.3e}, "
                f"RC={est.RC:.3e}, Rdson={est.Rdson:.3e}, "
                f"Rload1={est.Rload1:.3e}, Rload2={est.Rload2:.3e}, "
                f"Rload3={est.Rload3:.3e}, Vin={est.Vin:.3f}, VF={est.VF:.3e}",
            )

    # Save the history to a CSV file
    training_run = TrainingRun.from_histories(
        loss_history=history_loss,
        param_history=history_params,
    )

    # generate the output directory if it doesn't exist
    db_dir.mkdir(parents=True, exist_ok=True)

    # if savename doesn't end with .csv, add it
    if not savename.endswith(".csv"):
        savename += ".csv"

    training_run.save_to_csv(db_dir / savename)
    print("Concluded training.")

# %%
set_seed(123)
device = "cpu"

# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"

GROUP_NUMBER_DICT = {
    0: "ideal",
    1: "ADC_error",
    3: "5 noise",
    4: "10 noise",
}


lr = 1e-3
epochs = 20_000
patience = 5000
device = "cpu"  # or "cuda" if you have a GPU
lr_reduction_factor = 0.5


out_dir = Path.cwd().parent / "RESULTS" / "Bayesian" / "Adam"

# noisy_measurements = {}
# for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
#     if "Sync" in group_name:
#         # Skip the Sync Error group for now
#         continue
#     print(f"Loading group {group_number}: {group_name}")
#     # Load the data from the hdf5 file
#     io = LoaderH5(db_dir, h5filename)
#     io.load(group_name)

#     # Store the measurement in a dictionary
#     noisy_measurements[group_name] = io.M

#     print(f"\n{'-'*50}")
#     print(f"{idx}) Training with {group_name} data")

#     # Train the model on the noisy measurement
#     train_from_measurement_file(
#         io.M,
#         initial_guess_params= NOMINAL,
#         db_dir=out_dir,
#         savename=f"noisy_run_{group_name}.csv",
#         lr=lr,
#         patience=patience,
#         lr_reduction_factor=lr_reduction_factor,
#         epochs=epochs_1,
#         device=device,
#     )

# %%
from pinn_buck.plot_utils import (
    plot_tracked_parameters,
    plot_final_percentage_error,
    plot_final_percentage_error_multi,
)

# %% [markdown]
# ## MAP Estimation

# %% [markdown]
# In order to get a MAP estimation, we have to consider the priors and try to maximize the posterior distribution on the parameters following Bayes' rule:
# $$ p(z | x = D) = p(x=D | z) \cdot p(z) / p(x=D) $$
# However, even in this low dimensional case with 10 latent parameters $z$, which are the physical parameters of the circuit, it is difficult to numerically compute the marginal $$p(x=D) = \int_z p(x=D | z) p(z) dz$$
#
# However, we know that the marginal does not depend on the particular choice of z, so the level set for fixed x=D of the joint $p(x=D, z) = p(x=D | z) \cdot p(z)$ is proportional to the posterior:
# $$ p(z | x = D) \propto  p(x=D, z)$$
# Then it is possible to get the MAP solely by analyzing the joint distribution since:
# $$ \argmax_z p(z | x=D) = \argmax_z p(x=D, z)$$
#
# ### Definition of the loss function
#
# Now we can elaborate on the expression of the MAP estimate, that searches for $z_{\text{MAP}}=\argmax_z p(x=D, z)$. To this end, we chose the Negative Log Likelihood as a loss function
#
# \begin{align}
#     \text{NLL}(z) &= - \log p(x=D, z) \\
#     &= -\log p(x=D | z) - \log p(z)
# \end{align}
#
# With the hypothesis of independent draws of the observed variables x that form the dataset:
# $$ p(x=D | z) = \prod_{i=0}^{N-1} p(x_i | z) $$
#
#
# ## Consider the choice of Distribution
#
# We have chosen a Log-Normal distribution to represent the pdf over the circuit parameters:
# $$p(x) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left( -\frac{(\log x - \mu)^2}{2\sigma^2} \right)$$
#
# or equivalently $\log x \sim \mathcal{N}(x; \mu, \sigma)$:
#
# $$p(\log x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\log x - \mu)^2}{2\sigma^2}\right)$$
#
# Then:
# \begin{align}
#     \log p(\log z) &= \log \left[\frac{1}{\sqrt{2\pi\sigma_0^2}} \exp\left(-\frac{(\log z - \mu_0)^2}{2\sigma_0^2}\right)\right]\\
#                 & \propto -\frac{(\log z - \mu_0)^2}{2\sigma_0^2}
# \end{align}
#
#
# Assuming the **observations** $\{x_i\}_{i=0}^{N-1}$ are **independent and Gaussian distributed** with a likelihood of the form $x_i \sim \mathcal{N}(f(z), \sigma_l^2)$, the total log-likelihood is:
#
# $$
# \log p(\mathbf{x} \mid z) = \sum_{i=0}^{N-1} \log p(x_i \mid z)
# \propto -\frac{1}{2\sigma_l^2} \sum_{i=0}^{N-1} (x_i - f(z))^2
# $$
#
# So the nnegative log likelihood becomes:
# $$ \text{NLL}(z) = \frac{1}{2\sigma_l^2} \sum_{i=0}^{N-1} (x_i - f(z))^2 + \frac{(\log z - \mu_0)^2}{2\sigma_0^2} + \text{const.}$$
#
# Note that we assume a fixed variance of the observation that doesn't depend on the values of the input parameters, i.e. we assume **homoschedasticity**. Note that this makes sense:
#
# >  "The parameters determine the mean values of the observed data with some random noise whose variance depends on random factors, independent on the data."
#
# Many times, the prior assumptions are weak and therefore, therefore $\sigma_0$ is allowed to control the data-prior balance entirely. However, in our case we are considering different solutions with different noise powers. Therefore, we should consider both $\sigma_l$ and $\sigma_0$ explicitly. We can rewrite the NLL as:
#
# $$\text{NLL}(z) = \frac{1}{2\sigma_l^2} \left[\sum_{i=0}^{N-1} (x_i - f(z))^2 + \frac{\sigma_l^2}{\sigma_0^2}\cdot(\log z - \mu_0)^2 \right] + \text{const.}$$
#
# Since $\frac{1}{2\sigma_l^2}$ only changes the scale of the NLL we can ignore this term to obtain:
#
# $$\text{NLL}(z) = \sum_{i=0}^{N-1} (x_i - f(z))^2 + \left(\frac{\sigma_l}{\sigma_0}\right)^2\cdot(\log z - \mu_0)^2 + \text{const.}$$
#
# Note that, if the data is noisy, then the prior has a stronger influence on the loss, while if the data noise is very low, we quickly move away from our prior convictions.
#
# In case, you don't have a clear idea of the noise power, you can leave $\sigma_l = 1$ and model the prior uncertainty via the prior $\sigma_0$ alone.

# %%
from typing import Callable
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from contextlib import contextmanager

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
        Rload1=_to_sigma(rel_tol.Rload1),
        Rload2=_to_sigma(rel_tol.Rload2),
        Rload3=_to_sigma(rel_tol.Rload3),
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
        proposed_value = getattr(logparams, name)
        mu = getattr(nominal_logparams, name)
        sig = getattr(sigma, name)
        total += ((proposed_value - mu) / sig) ** 2
    return total


# define the loss function for MAP estimation that combines the L2 loss and the log-normal prior.
def make_map_loss(nominal: Parameters, sigma: Parameters, data_noise: float = 1) -> Callable:
    """Return a closure that adds the prior term."""

    def _loss(model, preds, y_n, y_np1):
        return l2_loss(preds, y_n, y_np1) + data_noise * log_normal_prior(model.logparams, nominal, sigma)

    return _loss


# # --- small helper so we can temporarily overwrite model params ------------
# @contextmanager
# def _temporary_params(model, vec):
#     """Temporarily copy `vec` into model.log_* parameters."""
#     # stash original
#     original = parameters_to_vector(
#         [getattr(model, f"log_{n}") for n in Parameters._fields]
#     ).clone()
#     try:
#         vector_to_parameters(vec, [getattr(model, f"log_{n}") for n in Parameters._fields])
#         yield
#     finally:
#         # restore original weights
#         vector_to_parameters(original, [getattr(model, f"log_{n}") for n in Parameters._fields])


# # --------------------------------------------------------------------------
# def laplace_posterior(model, X, y, y_prev, loss_fn, damping=1e-6):
#     """
#     Return (theta_map, Sigma_log) where
#       theta_map  : flattened 10-vector of log(S·θ) at MAP
#       Sigma_log  : 10×10 covariance in the same space
#     """
#     # 1) Flatten current log-params (these are your MAP weights)
#     theta_map = (
#         parameters_to_vector([getattr(model, f"log_{n}") for n in Parameters._fields])
#         .detach()
#         .clone()
#     )

#     # 2) Define a pure function f(vec) = loss with vec injected
#     def f(vec):
#         with _temporary_params(model, vec):
#             preds = model(X, y)
#             return loss_fn(model, preds, y_prev, y)

#     # 3) Hessian in log-space
#     H = torch.autograd.functional.hessian(f, theta_map)

#     # 4) Damped inverse
#     eye = torch.eye(len(theta_map), device=theta_map.device, dtype=theta_map.dtype)
#     Sigma_log = torch.linalg.inv(H + damping * eye)

#     return theta_map, Sigma_log

# %%
from dataclasses import dataclass

# for simplicity let's define a dataclass for the training configurations
@dataclass
class AdamOptTrainingConfigs:
    savename: str = "saved_run"
    out_dir: Path = Path(".")
    lr: float = 1e-3
    epochs: int = 30_000
    epochs_lbfgs: int = 1500
    device: str = "cpu"
    patience: int = 5000
    lr_reduction_factor: float = 0.5


# Define the Trainer class for training the model using Adam.
class Trainer:
    def __init__(self, model, loss_fn: Callable, optim_cfg: AdamOptTrainingConfigs, device="cpu"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optim_cfg = optim_cfg
        self.device = device
        self.history = {"loss": [], "params": [], "lr": []}

    def fit(self, X, y, y_prev, epochs_adam=20_000, epochs_lbfgs=500, callback=None):

        X = X.detach().to(self.device)
        y = y.detach().to(self.device)
        y_prev = y_prev.detach().to(self.device)

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

        for it in range(epochs_adam):
            opt.zero_grad()
            loss: torch.Tensor = self.loss_fn(self.model, self.model(X, y), y_prev, y)
            loss.backward()

            opt.step()
            scheduler.step(loss.item())

            # print("\nGradient check for log-parameters:")
            # for name in Parameters._fields:
            #     param = getattr(self.model, f"log_{name}")
            #     if param.grad is None:
            #         print(f"  log_{name}: grad = None ❌")
            #     else:
            #         norm = param.grad.norm().item()
            #         print(f"  log_{name}: grad norm = {norm:.3e} {'✅' if norm > 0 else '⚠️ zero'}")

            if it % 1000 == 0:
                est = self.model.get_estimates()
                self.history["loss"].append(loss.item())
                self.history["params"].append(est)
                # record the learning rate
                self.history["lr"].append(opt.param_groups[0]["lr"])

                if loss.item() < best_loss:
                    best_loss = loss.item()

                # print the parameter estimation
                grads = [getattr(self.model, f"log_{n}").grad.view(1) for n in Parameters._fields if getattr(self.model, f"log_{n}").grad is not None]

                if grads:
                    gradient_vector = torch.cat(grads)
                    gradient_norm = gradient_vector.norm().item()
                else:
                    gradient_norm = float('nan')  # no gradients found (shouldn't happen during training)

                print(
                    f"Iteration {it}, gradient_norm {gradient_norm:4e}, loss {loss:4e},  Parameters (Adam):",
                    f"L={est.L:.3e}, RL={est.RL:.3e}, C={est.C:.3e}, "
                    f"RC={est.RC:.3e}, Rdson={est.Rdson:.3e}, "
                    f"Rload1={est.Rload1:.3e}, Rload2={est.Rload2:.3e}, "
                    f"Rload3={est.Rload3:.3e}, Vin={est.Vin:.3f}, VF={est.VF:.3e}",
                )

        # # → LBFGS
        # LBFGS optimization tends to find stable solutions that also minimize the gradient norm.
        # This will be useful when we want to compute the Laplace posterior, which relies on the Hessian of the loss function.

        print("Starting LBFGS optimization...")
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=epochs_lbfgs,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-8,
        )

        # update the histories with the last Adam iteration
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
        print("Concluded ADAM training.")

        def closure():
            opt.zero_grad()
            loss = self.loss_fn(self.model, self.model(X, y), y_prev, y)
            loss.backward()
            return loss

        opt.step(closure)

        # # calculate the Laplace posterior
        # theta_map, cov_log = laplace_posterior(
        # self.model, X, y, y_prev, self.loss_fn, damping=1e-6)

        # self.theta_map = theta_map      # save for later inspection
        # self.cov_log   = cov_log

        return self.model

# %%
# get data noise values

# ADC noise is 1 LSB, so we can assume that the data noise is 1 LSB
# 5 noise is 5 LSB, 10 noise is 10 LSB

# A least significant bit (LSB) is the smallest unit of data in a digital system and is calculated as:
#         LSB_i = I_FS / (2**12 - 1)  # assuming 12-bit ADC
#         LSB_v = V_FS / (2**12 - 1)  # assuming 12-bit ADC

# where I_FS is the full-scale current, set to 10 A and V_FS is the full-scale voltage set to 30 V (see 03_inspect_noisy_data.ipynb).

# Then the noise level is calculated as:
#         noise_level_i = noise_level * LSB_i  # normalize noise level to LSB
#         noise_level_v = noise_level * LSB_v  # normalize noise level to LSB

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

# In order to use the noise power in the loss function, we need to convert it to the same scale as the model.
# In the model, the data is rescaled with
#
#       def _scale(self, x: torch.Tensor):
#           return 2 * (x - self.lb) / (self.ub - self.lb) - 1
# So we need to scale the noise power accordingly: SCALE = 2 / self.ub - self.lb
# and the variance of the noise in the model is then:
# noise_power_norm = noise_power * SCALE**2


def noise_power_rescaler(
    noise_power_i: float, noise_power_v: float, X: torch.Tensor
) -> float:
    """Rescale the noise power to the model scale."""
    i = X[:, 0]  # current
    v = X[:, 1]  # voltage
    
    i_max, i_min = i.max(), i.min()
    v_max, v_min = v.max(), v.min()
    
    SCALE_i = 2 / (i_max - i_min)  # scale for current
    SCALE_v = 2 / (v_max - v_min)  # scale for voltage
    
    # average i jump and v jump
    i_jump = (torch.diff(i)**2).mean().item() * SCALE_i**2
    v_jump = (torch.diff(v)**2).mean().item() * SCALE_v**2
    
    return (noise_power_i / i_jump + noise_power_v  / v_jump) / 2


# %%
out_dir = Path.cwd().parent / "RESULTS" / "Bayesian" / "Adam_MAP"

run_configs = AdamOptTrainingConfigs(
    savename="adam_run.csv",
    out_dir=out_dir,
    lr=lr,
    epochs=epochs,
    epochs_lbfgs=5000,
    device=device,
    patience=patience,
    lr_reduction_factor=lr_reduction_factor,
)


# load the transient data as unified numpy arrays
def load_data_to_model(meas: Measurement, initial_guess_params: Parameters):
    """Load the data from a Measurement object and return the model."""
    # load the transient data as unified numpy arrays
    X, y = meas.data
    s1, s2, s3 = list(
        map(lambda x: x - 1, meas.transient_lengths)
    )  # subtract 1 since we use the previous time step as input
    lb, ub = X.min(0), X.max(0)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    # Model
    model = BuckParamEstimator(lb, ub, s1, s2, s3, initial_guess_params).to(device)
    return X_t, y_t, model


noise_power_dict = {
    0: (1e-12, 1e-12),  # ideal
    1: (noise_power_ADC_i, noise_power_ADC_v),  # ADC error
    3: (noise_power_5_i, noise_power_5_v),  # 5 noise
    4: (noise_power_10_i, noise_power_10_v),  # 10 noise
}

noisy_measurements = {}
for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    # Load the data from the hdf5 file
    io = LoaderH5(db_dir, h5filename)
    io.load(group_name)

    # Store the measurement in a dictionary
    noisy_measurements[group_name] = io.M

    run_configs.savename = f"noisy_run_{group_name}.csv"

    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")

    # Train the model on the noisy measurement
    X, y, model = load_data_to_model(
        meas=io.M,
        initial_guess_params=NOMINAL,
    )

    prior_info = {
        "nominal": NOMINAL,
        "sigma": rel_tolerance_to_sigma(REL_TOL),
    }

    trainer = Trainer(
        model=model,
        loss_fn=make_map_loss(
            **prior_info, data_noise=noise_power_rescaler(*noise_power_dict[group_number], X.clone().detach())
        ),
        optim_cfg=run_configs,
        device=device,
    )

    trainer.fit(
        X=X,
        y=y,
        y_prev=X[:, :2],  # use the first two columns as previous time step
        epochs_adam=run_configs.epochs,
    )

    print("\n \n \n")

# %%
from typing import Dict


from pinn_buck.plot_utils import (
    plot_tracked_parameters,
    plot_final_percentage_error,
    plot_final_percentage_error_multi,
)

# loop through all CSV files in the directory
csv_files = list(out_dir.glob("*.csv"))
runs = {}
for csv_file in csv_files:
    print(f"Processing {csv_file.name}")
    # noise_level = float(csv_file.stem.split("_")[-1])  # Extract noise level from filename
    # tr = TrainingRun.from_csv(csv_file)
    # runs[f"{noise_level}*LSB"] = tr.drop_columns(["learning_rate"])  # drop learning rate column

    label = csv_file.stem.removeprefix("noisy_run_")  # Extract label from filename
    tr = TrainingRun.from_csv(csv_file)
    runs[label] = tr.drop_columns(["learning_rate"])  # drop learning rate column

GROUP_NUMBER_DICT = {
    0: "ideal",
    1: "ADC_error",
    2: "Sync Error",
    3: "5 noise",
    4: "10 noise",
    5: "ADC-Sync-5noise",
    6: "ADC-Sync-10noise",
}


for ii in (0, 1, 3, 4):
    tr = runs[GROUP_NUMBER_DICT[ii]]
    label = GROUP_NUMBER_DICT[ii]
    if ii == 0:
        fig, ax = plot_tracked_parameters(
            df=tr,
            target=TRUE_PARAMS,
            label=label,
            color="black",
            figsize=(18, 10),
        )
        continue

    plot_tracked_parameters(df=tr, target=None, label=label, ax=ax, color=None)

runs_ordered: Dict[str, TrainingRun] = {GROUP_NUMBER_DICT[ii]: runs[GROUP_NUMBER_DICT[ii]] for ii in (0, 1, 3, 4)}

plot_final_percentage_error_multi(
    runs=runs_ordered, target=TRUE_PARAMS, figsize=(14, 5), select_lowest_loss=False
)

# %%
import matplotlib.pyplot as plt

# Plotting
param_names = Parameters._fields
ncols = 2
nrows = int(np.ceil(len(param_names) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))
axes = axes.flatten()


for i, name in enumerate(param_names):
    nominal = getattr(NOMINAL, name)
    rel_tol = getattr(REL_TOL, name)
    true_val = getattr(TRUE_PARAMS, name)
    estimations_for_different_runs: Dict[str, Parameters] = {
        label: getattr(run.best_parameters, name) for label, run in runs_ordered.items()
    }

    # log-normal parameters
    sigma = np.log(1 + rel_tol)
    mu = np.log(nominal)

    dist = lognorm(s=sigma, scale=np.exp(mu))

    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
    pdf = dist.pdf(x)

    ax = axes[i]
    ax.plot(x, pdf, label=f"{name} prior")
    ax.axvline(true_val, color="red", linestyle="--", label="TRUE")
    
    # plot the nominal value
    ax.axvline(nominal, color="green", linestyle="--", label="Nominal")
    
    for label, est in estimations_for_different_runs.items():
        ax.axvline(est, linestyle=":", label=f"{label} estimate")
    ax.set_title(name)
    ax.set_yticks([])
    ax.legend()

    if name == "L":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
        ax.set_xlabel("[mH]")
    if name == "C":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
        ax.set_xlabel("[μF]")

fig.suptitle("Log-Normal Priors (linear space) with TRUE values", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.95)

# %%
trainer.model

# %%


# %%


# %%
# 1. Run Laplace approx
theta_map, Sigma_log = laplace_posterior(
    trainer.model,
    X,
    y,
    X[:, :2],
    make_map_loss(nominal=NOMINAL, sigma=rel_tolerance_to_sigma(REL_TOL)),
    damping=1e-3,
)

# %%
Sigma_log

# %%
num_samples = 1000
samples_log = torch.distributions.MultivariateNormal(theta_map, Sigma_log).sample(
    (num_samples,)
)  # shape: (1000, 10)


# 3. Convert samples to physical units
def to_physical(name, log_sample):
    return torch.exp(log_sample) / _SCALE[name]


phys_samples = {
    name: to_physical(name, samples_log[:, i]) for i, name in enumerate(Parameters._fields)
}

# %%
# plot the samples_log
fig, axes = plt.subplots(5, 2, figsize=(12, 12), constrained_layout=True)
axes = axes.flatten()

for i, name in enumerate(Parameters._fields):
    ax = axes[i]
    ax.hist(samples_log[:, i].numpy(), bins=50, density=True, alpha=0.5, label=f"{name} samples")
    ax.axvline(getattr(TRUE_PARAMS, name), color="red", linestyle="--", label="TRUE")
    ax.axvline(getattr(NOMINAL, name), color="green", linestyle="--", label="Nominal")

    ax.set_title(name)
    ax.set_yticks([])
    ax.legend()

    if name == "L":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
        ax.set_xlabel("[mH]")
    if name == "C":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
        ax.set_xlabel("[μF]")

# %%
samples_log.shape

# %%
import scipy
# 4. Plot histograms of marginals
ncols = 2
nrows = int(np.ceil(len(phys_samples) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
axes = axes.flatten()

for i, name in enumerate(Parameters._fields):
    ax = axes[i]
    vals = phys_samples[name].numpy()

    ax.hist(vals, bins=40, density=True, alpha=0.6, label="Laplace posterior")

    # Overlay Gaussian PDF (optional)
    mu = np.mean(vals)
    sigma = np.std(vals)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax.plot(x, scipy.stats.norm.pdf(x, loc=mu, scale=sigma), label="Gaussian fit", linestyle="--")

    # Add MAP point and true value (if available)
    map_phys = to_physical(name, theta_map[i]).item()
    ax.axvline(map_phys, color="black", linestyle="-", label="MAP")

    if "TRUE" in globals():
        ax.axvline(getattr(TRUE_PARAMS, name), color="red", linestyle="--", label="TRUE")

    ax.set_title(name)
    ax.legend()

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.tight_layout()
fig.suptitle("Laplace Posterior Marginals (Physical Units)", fontsize=16)
fig.subplots_adjust(top=0.95)
plt.show()

# %%
group_name = GROUP_NUMBER_DICT[0]

io = LoaderH5(db_dir, h5filename)
io.load(group_name)

# Store the measurement in a dictionary
noisy_measurements[group_name] = io.M

run_configs.savename = f"noisy_run_{group_name}.csv"    

print(f"\n{'-'*50}")
print(f"{idx}) Training with {group_name} data")

# Train the model on the noisy measurement
X, y, model = load_data_to_model(
    meas=io.M,
    initial_guess_params=NOMINAL,
)

run_configs = AdamOptTrainingConfigs(
    savename="adam_run.csv",
    out_dir=out_dir,
    lr=lr,
    epochs=9000,
    device=device,
    patience=patience,
    lr_reduction_factor=lr_reduction_factor,
)
trainer = Trainer(
    model=model,
    loss_fn=make_map_loss(nominal=NOMINAL, sigma=rel_tolerance_to_sigma(REL_TOL)),
    optim_cfg=run_configs,
    device=device,
)

trainer.fit(
    X=X,
    y=y,
    y_prev=X[:, :2],  # use the first two columns as previous time step
    epochs_adam=run_configs.epochs,
)

# %%
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# my current estimates in physical units
est = model.get_estimates()
print(est)


def flat_logparams(model):
    return torch.cat([p.view(1) for p in model.logparams])
theta_map = flat_logparams(model).detach().clone()

print("MAP (log-space):", theta_map)
print("MAP (physical units):")
print(model.get_estimates())


# %%
from torch.func import functional_call

model = trainer.model


# my current estimates in physical units
est = model.get_estimates()
print(est)


def flat_logparams(model):
    return torch.cat([p.view(1) for p in model.logparams])


theta_map = flat_logparams(model).detach().clone().requires_grad_()

print("MAP (log-space):", theta_map)
print("MAP (physical units):")
print(model.get_estimates())


y = y.detach().to(device)
y_prev = X[:, :2].detach().to(device)  # use the first two columns as previous time step

def _loss(model, preds, y_n, y_np1):
    return l2_loss(preds, y_n, y_np1) + log_normal_prior(model.logparams, NOMINAL, rel_tolerance_to_sigma(REL_TOL))


def loss_on_params(theta_vec):
    # Map vector → log_parameters dict and Parameters tuple
    param_keys = [f"log_{n}" for n in Parameters._fields]
    split_vals = [v.view_as(getattr(model, k)) for k, v in zip(param_keys, theta_vec)]

    # Create the state dict for functional_call
    new_params = {k: v for k, v in zip(param_keys, split_vals)}

    # Create a Parameters object for log_prior evaluation
    logparams = Parameters(*split_vals)  # same order as Parameters._fields

    # Forward + loss
    preds = functional_call(model, new_params, (X, y))
    return l2_loss(preds, y_prev, y) + log_normal_prior(
        logparams, NOMINAL, rel_tolerance_to_sigma(REL_TOL)
    )


loss = loss_on_params(theta_map)
print("Loss at MAP:", loss.item())

# Backpropagate
loss.backward()

# Now it works:
print("Gradient norm at MAP:", theta_map.grad.norm().item())

# %%
from torch.autograd.functional import hessian

H = hessian(loss_on_params, theta_map)

print("Hessian shape:", H.shape)  # should be [10, 10]
eigs = torch.linalg.eigvalsh(H)
print("Eigenvalues:", eigs)

# %%
damping = 1e-6 * torch.eye(H.shape[0], device=H.device)
Sigma_log = torch.linalg.inv(H + damping)

print("Covariance shape:", Sigma_log.shape)  # should be [10, 10]
print("Covariance log matrix eigenvalues:", torch.linalg.eigvalsh(Sigma_log))


# tranform to physical units
params_phys = model.get_estimates()
theta_map_phys = torch.tensor([
    getattr(params_phys, name) for name in Parameters._fields
])

# Convert covariance to physical units
jacobian = torch.diag(theta_map_phys)
sigma_phys = jacobian @ Sigma_log @ jacobian.T
print("Covariance in physical units shape:", sigma_phys.shape)  # should be [10, 10]
print("Covariance in physical units eigenvalues:", torch.linalg.eigvalsh(sigma_phys))


# %%
std_phys = torch.sqrt(torch.diag(sigma_phys))
for i, name in enumerate(Parameters._fields):
    mean = theta_map_phys[i].item()
    std = std_phys[i].item()
    print(f"{name:8s}: {mean:.3e} ± {std:.1e}  ({100*std/mean:.2f} %)")

# %%
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
import numpy as np

# Plotting
param_names = Parameters._fields
ncols = 2
nrows = int(np.ceil(len(param_names) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 14))
axes = axes.flatten()

# Extract posterior mean and std from Laplace approximation
theta_map_np = theta_map_phys.cpu().numpy()
std_map_np = np.sqrt(torch.diag(sigma_phys).cpu().numpy())

mu_log_np = theta_map.cpu().detach().numpy()  # 10-vector, log space MAP
sigma_log_np = np.sqrt(torch.diag(Sigma_log).cpu())  # 1-sigma (log space)

for i, name in enumerate(param_names):
    nominal = getattr(NOMINAL, name)
    rel_tol = getattr(REL_TOL, name)
    true_val = getattr(TRUE_PARAMS, name)
    mu_post = theta_map_np[i]
    std_post = std_map_np[i]

    # --- prior: log-normal ---
    sigma_prior = np.log(1 + rel_tol)
    mu_prior = np.log(nominal)
    prior_dist = lognorm(s=sigma_prior, scale=np.exp(mu_prior))
    x_prior = np.linspace(prior_dist.ppf(0.001), prior_dist.ppf(0.999), 500)
    y_prior = prior_dist.pdf(x_prior)

    # --- posterior: Gaussian ---
    x_post_gauss = np.linspace(mu_post - 4 * std_post, mu_post + 4 * std_post, 500)
    y_post_gauss = norm(loc=mu_post, scale=std_post).pdf(x_post_gauss)

    mu_log = mu_log_np[i] - np.log(_SCALE[name])       # remove scale so it's log θ
    sigma_log = sigma_log_np[i]

    post_dist = lognorm(s=sigma_log, scale=np.exp(mu_log))

    x_post = np.linspace(post_dist.ppf(0.001), post_dist.ppf(0.999), 500)
    y_post = post_dist.pdf(x_post)    

    ax = axes[i]
    ax.plot(x_prior, y_prior, label="Prior (log-normal)", color="blue", linewidth=1)
    ax.plot(x_post_gauss, y_post_gauss, label="Laplace Posterior (Gauss Approx.)", color="orange", linewidth=1)
    ax.plot(x_post, y_post, label="Laplace Posterior (log-normal)", color="black", linewidth=2)

    # markers
    ax.axvline(true_val, color="red", linestyle="--", label="TRUE")
    ax.axvline(nominal, color="green", linestyle="--", label="Nominal")

    # point estimates from other runs
    estimations_for_different_runs: Dict[str, Parameters] = {
        label: getattr(run.best_parameters, name) for label, run in runs_ordered.items()
    }
    for label, est in estimations_for_different_runs.items():
        ax.axvline(est, linestyle=":", label=f"{label} estimate")

    ax.set_title(name)
    ax.set_yticks([])

    # Label formatting
    if name == "L":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
        ax.set_xlabel("[mH]")
    elif name == "C":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
        ax.set_xlabel("[μF]")

    ax.legend(fontsize="x-small", loc="upper right")

fig.suptitle("Prior (log-normal) and Laplace Posterior for Each Parameter", fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

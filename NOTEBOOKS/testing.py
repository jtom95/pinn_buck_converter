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
# ## Prior Distribution Family over the Physical Parameters

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
from pinn_buck.model.loss_function_configs import l2_loss
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
    print("Concluded ADAM training.")

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
epochs = 30_000
patience = 5000
device = "cpu"  # or "cuda" if you have a GPU
lr_reduction_factor = 0.5


out_dir = Path.cwd() / "RESULTS" / "Bayesian" / "Adam"

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
# In this application we have a supervised model, so
# $$y = f(x, z)$$
# And he dataset is $\mathcal D=\{(x_i,\,y_i)\}_{i=0}^{N-1}$.
#
# We can rewrite:
# $$ p(z | x, y = D) \propto  p(y=D_y, z | x=D_x) =  p(y = D_y \mid x, z) p(z)$$
#
# ### Definition of the loss function
# Now we can elaborate on the expression of the MAP estimate, that searches for $z_{\text{MAP}}=\argmax_z p(y=D, z | x)$. To this end, we chose the Negative Log Likelihood as a loss function
#
# \begin{align}
#     \text{NLL}(z) &= - \log p(y=D, z | x) \\
#     &= -\log p(y=D | z, x) - \log p(z)
# \end{align}
#
# With the hypothesis of independent draws of the observed variables x that form the dataset:
# $$ p(y=D | z, x) = \prod_{i=0}^{N-1} p(y_i | x_i, z) $$
#
#
# ## Consider the choice of Distribution
#
# ### Prior
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
# Assuming the priors are independently drawn:
# $$
# \log p(\log z)= -\frac{1}{2}\sum_{k=1}^{10}
#       \frac{\bigl(z_k-\mu_{0,k}\bigr)^{2}}{\sigma_{0,k}^{2}}+\text{const}.
# $$
#
#
#
# ### Likelihood
# Assuming the **observations** $\{x_i\}_{i=0}^{N-1}$ are  corrupted by additive homoscedastic Gaussian sensor noise, so the observation random variables are **independent and Gaussian distributed** with a likelihood of the form $y_i \mid x_i,z \;\sim\; \mathcal N\!\bigl(f(x_i,z),\,\sigma_x^{2}\mathbf I\bigr)$, the total log-likelihood is:
#
# $$
# \log p(\mathbf y\mid\mathbf x,z)
#   = -\frac{1}{2\sigma_x^{2}}\sum_{i=0}^{N-1}\!
#       \bigl\|\,y_i-f(x_i,z)\bigr\|^{2} + \text{const}.
# $$
# Note that $\sigma_x$ is the noise in the observed dataset. However, it can be useful to instead consider the distribution of residuals $r_n = y - f(x,z)$, which, if for now we consider $f(x,z)$ deterministic, is $r_n \sim \mathcal{N}(0, \sigma_x^{2}\mathbf I)$, i.e. **it also exhibits an uncertainty derived from the data noise $\sigma_x$**.
#
# ### NLL - MAP objective
# So the nnegative log likelihood becomes:
# $$ \text{NLL}(z)=
# \frac{1}{2\sigma_x^{2}}
#       \sum_{i=0}^{N-1}\!\bigl\|y_i-f(x_i,z)\bigr\|^{2}
# +\frac{1}{2}\sum_{k=1}^{10}
#       \frac{\bigl(\log z_k-\mu_{0,k}\bigr)^{2}}{\sigma_{0,k}^{2}}
# + \text{const}.
# $$
#
# Note that we assume a fixed variance of the observation that doesn't depend on the values of the input parameters, i.e. we assume **homoschedasticity**. Note that this makes sense:
#
# >  "The parameters determine the mean values of the observed data with some random noise whose variance depends on random factors, independent on the data."
#
# Many times, the prior assumptions are weak and therefore, therefore $\sigma_0$ is allowed to control the data-prior balance entirely. However, in our case we are considering different solutions with different noise powers. Therefore, we should consider both $\sigma_x$ and $\sigma_0$ explicitly. We can rewrite the NLL as:
#
# $$\text{NLL}(z) = \frac{1}{2\sigma_x^2} \left[
#       \sum_{i=0}^{N-1}\!\bigl\|y_i-f(x_i,z)\bigr\|^{2}
# + \frac{\sigma_x^2}{\sigma_0^2} \sum_{k=1}^{10}
#       \bigl(\log z_k-\mu_{0,k}\bigr)^{2}
# \right] + \text{const.}$$
#
# Since $\frac{1}{2\sigma_x^2}$ only changes the scale of the NLL we can ignore this term to obtain:
#
# $$\text{NLL}(z) = \sum_{i=0}^{N-1}\!\bigl\|y_i-f(x_i,z)\bigr\|^{2}
# + \underbrace{\frac{\sigma_x^2}{\sigma_0^2}}_\lambda \sum_{k=1}^{10}
#       \bigl(\log z_k-\mu_{0,k}\bigr)^{2} + \text{const.}$$
#
# Here we can set $\frac{\sigma_x^2}{\sigma_0^2}=\lambda$ and we can see that we obtain a new parameter that regulates how much weights we want to give to the prior. Indeed, if the data is noisy, then $\lambda \uparrow$ and the prior has a stronger influence on the loss, while if the data noise is very low, $\lambda \downarrow$ and we quickly move away from our prior convictions.
#
# It is common in these setups to directly tune $\lambda$ as a hyperparameter, for example via cross-validation and grid search. However, in our case there is a strong physical interpretation for these quantities and we may have a meaningful insite on the prior range on the parameters from datasheets or manufactorers. Moreover, we may also have some knowledge on the noise present in our measurements. Therefore, we can maintain the original expression for the posterior loss:
#
# $$\text{NLL}(z)=
# \frac{1}{2\sigma_x^{2}}
#       \sum_{i=0}^{N-1}\!\bigl\|y_i-f(x_i,z)\bigr\|^{2}
# +\frac{1}{2}\sum_{k=1}^{10}
#       \frac{\bigl(\log z_k-\mu_{0,k}\bigr)^{2}}{\sigma_{0,k}^{2}}
# + \text{const}.$$

# %% [markdown]
#
# ## Noisy Observations
#
# In the previously described model, all observation noise was attributed to the target variable:
#
# $$
# y \sim \mathcal{N}(f(x, z), \sigma_x^2).
# $$
#
# However, in our specific setup, the target $y = x_{n+1}$ is itself an observed state, and so is the input $x = x_n$. Both are corrupted by measurement noise of the same variance:
#
# $$
# x_n^{\text{obs}} = \tilde{x}_n + \eta_n, \quad \eta_n \sim \mathcal{N}(0, \sigma_x^2),
# $$
#
# $$
# y_n^{\text{obs}} = x_{n+1}^{\text{obs}}.
# $$
#
# We use a deterministic model $f(z, x_n)$, typically an RK4 integrator, to predict the next state. The residuals between the noisy target and the prediction are:
#
# $$
# r_n(z) = y_n^{\text{obs}} - f(z, x_n^{\text{obs}}).
# $$
#
# If noise levels were unknown, we could estimate the total predictive variance directly from the residuals:
#
# $$
# \hat{\sigma}_{\text{tot}}^2 = \frac{1}{N} \sum_{n=1}^N \left( r_n^{\text{nom}} - \bar{r}^{\text{nom}} \right)^2,
# $$
#
# where the residuals are computed using the nominal parameters $z_{\text{nom}}$. This is effectively empirical risk minimization and yields an approximate observation noise model, albeit biased by imperfect parameters.
#
# ---
#
# ### Analytical Estimation (When Noise is Known)
# We have a noisy observation $x^{\text{obs}}\!\sim\!\mathcal N(x,\sigma_x^2)$ and the forward model is a scalar
# $y^{\text{obs}}=f(x^{\text{obs}})$.
#
# Let us consider the residual we want to minimize:
# $r_n^{\text{obs}} = x_{n}^{\text{pred}}-x_{n}^{\text{obs}}$
#
# Then we see that:
# $$x_{n}^{\text{pred}} = f(x_{n-1}^{\text{obs}})$$
#
#
# We can approximate the function with the first order Taylor expansion:
# \begin{align}
# f(x_{n-1}^{\text{obs}}) &= f(x_{n-1} + \varepsilon_{n-1}) \\
# &\simeq f(x_{n-1}) + \frac{\partial f}{\partial x} \varepsilon_{n-1}
# \end{align}
#
# Where $\frac{\partial f}{\partial x}$ is the Jacobian $J$
# Putting everything together we get:
# \begin{align}
# r_n^{\text{obs}} &= f(x_{n-1}) - x_n + J \varepsilon_{n-1} - \varepsilon_{n}\\
# &= r_{n} + \epsilon_{r}
# \end{align}
#
# So $r_n \sim \mathcal{N}(r_{n}, \Sigma_r)$.
#
# We can calculate the variance by looking at the random variable $\varepsilon_{r} = J \varepsilon_{n-1} - \varepsilon_{n}$. Since we assume the noise on the data is white gaussian noise, so $\varepsilon_{n-1}$ and \varepsilon_{n-1}$ are independent, we get:
#
# $$\operatorname{Var}[r_n^{\text{obs}}] = ||J||^2 \Sigma_x + \Sigma_x = (1+||J||_F^2) \Sigma_x$$
#
# In conclusion:
# $$
# \boxed{
#    r_n^{\text{obs}} \sim \mathcal{N}(r_{n}, (1+||J||_F^2)\Sigma_x)
# }
# $$
#
# This decomposition highlights that uncertainty propagates from $x_n$ through the model via its Jacobian with respect to $x$. Thus, even if the model is deterministic, input uncertainty affects the output distribution.
#
# #### Interpretation and Estimation of the Jacobian
#
# Importantly, the Jacobian, i.e. the derivative $\partial f / \partial x$, is **not** a time derivative like $(x_{n+1} - x_n) / dt$. It is a sensitivity derivative that answers the question:
#
# > *"How much would the prediction $f(z, x_n)$ change if I slightly perturbed the input $x_n$?"*
#
# This derivative captures how uncertainty in the current state propagates into uncertainty in the next predicted state.
#
# We then need to numerically estimate the Jacobian norm:
#
# $$
# \left\langle \|J(z_\text{nom})\|_F^2 \right\rangle_n,
# $$
#
# This can be done by looping through the training inputs $x_n$, computing the Jacobian with respect to the model input $x_n$ via autograd, and aggregating the squared norm of these matrices. Obviously, the derivatives depends also on the latent parameters $z$, which are unknown before training. However, we can use the nominal values to provide a good guess.
#
# This yields a scalar correction factor that adjusts the data loss scale in the MAP objective.

# %% [markdown]
# ## Independent v and i noises
#
# In practice the noise levels on v and i can be quite different. Moreover, the estimation of the voltage in the next location will depend on both previous voltage and current as well as the current in the next step will depend on both previous voltage and current. However, the sensitivity to one may be different from the sensitivity to the other. Therefore we have to consider the full 2x2 Jacobian.
#
#
# #### 2 .  Vector case with diagonal noise
#
# Let the measured state be
#
# $$
# x_n^{\text{obs}}=(i_n^{\text{obs}},\,v_n^{\text{obs}})^\top,
# \qquad
# y_n^{\text{obs}}=(i_{n+1}^{\text{obs}},\,v_{n+1}^{\text{obs}})^\top .
# $$
#
# Assume **independent** noises
#
# $$
# \Sigma_x=\mathrm{diag}\!\bigl(\sigma_i^{2},\;\sigma_v^{2}\bigr),\qquad
# \Sigma_y=\mathrm{diag}\!\bigl(\sigma_i^{2},\;\sigma_v^{2}\bigr),
# $$
#
# and let
#
# $$
# J=\frac{\partial f}{\partial x}
# =\begin{bmatrix}
# J_{11}&J_{12}\\[2pt]
# J_{21}&J_{22}
# \end{bmatrix}\in\mathbb R^{2\times2}.
# $$
#
# The residual covariance is
#
# $$
# \boxed{
# \Sigma_{\text{tot}}
#    =\Sigma_x + J\,\Sigma_x\,J^\top
# }
# $$
#

# %% [markdown]
# ## Time-Series Covariance
# If we were optimize with the previously derived formula, the results would greatly overestimate the prediction certainties. The reason is that until now we have assumed the residuals $r^\text{obs} \sim \mathcal{N}(r, \Sigma_r)$, so that the likelihood from the data can be calculated as:
# $$ \log
# \left[
# \prod_{i=1}^N p_i(r_i | x_i, z)
# \right] = ...  = \frac{1}{2}
#       \sum_{i=0}^{N-1}\! r_i \Sigma_r^{-1} r_i^\top
# $$
#
# However, we cannot say that the residuals are i.i.d. Indeed, the successive values of our **state vector**
#
# $$
# x_n = (i_n, v_n)^\top
# $$
#
# are **not independent and identically distributed (i.i.d.)**, but they evolve along a smooth trajectory, meaning $x_{n+1}$ is **highly correlated** with $x_n$.
#
# If we define the usual “level–residual”
#
# $$
# r_n = y_{n}^{\text{pred}} - y_{n}^{\text{obs}},
# $$
#
# and minimize its squared norm under a Gaussian likelihood, the optimizer implicitly treats each $r_n$ as **an independent data point**.
# This **overcounts** information, leading to artificially **narrow posteriors** and **underestimated uncertainties**.
#
# ---
#
# ### First-Differencing: A Classical Solution
#
# A well-known technique to reduce autocorrelation in time series is **first-differencing**:
#
# $$
# \Delta x_n = x_{n+1} - x_n.
# $$
#
# This is often used to make nonstationary or correlated series more like white noise (Box–Jenkins methodology, ARIMA modeling, etc.).
# We adopt the same idea at the residual level.
#
# ### Increment (Δ) Residual
#
# The model performs step-wise prediction:
#
# $$
# y_{n+1}^{\text{pred}} = f_z(x_n).
# $$
#
# So the natural quantity of interest is not just how accurate $y_{n+1}^{\text{pred}}$ is,
# but how well the model captures the **change** in state.
#
# Let us define:
#
# * The **predicted increment**:
#
#   $$
#   \Delta y^{\text{pred}} = y_{n+1}^{\text{pred}} - y_n^{\text{pred}},
#   $$
# * The **observed increment**:
#
#   $$
#   \Delta y^{\text{obs}} = y_{n+1}^{\text{obs}} - y_n^{\text{obs}}.
#   $$
#
# Then we define the **increment residual** as
#
# $$
# \boxed{
# r_\Delta = \Delta y^{\text{pred}} - \Delta y^{\text{obs}}
#          = (y_{n+1}^{\text{pred}} - y_n^{\text{pred}})
#          - (y_{n+1}^{\text{obs}} - y_n^{\text{obs}})
# }
# $$
#
# In terms of level-residuals,
#
# $$
# r_n = y_{n+1}^{\text{pred}} - y_{n+1}^{\text{obs}}, \quad
# r_{n-1} = y_n^{\text{pred}} - y_n^{\text{obs}},
# $$
#
# we get
#
# $$
# r_\Delta = r_n - r_{n-1}.
# $$
#
# ---
#
# ### Why the Δ–Residual Is Better
#
# * **De-correlation**:
#   Subtracting successive residuals eliminates the smooth, shared component of $r_n$ and $r_{n-1}$.
#   What remains is mostly sensor noise and model error, which are far more independent.
#   This matches the assumptions of the Gaussian likelihood far more closely.
#
# * **Effective Sample Size**:
#   Because residuals become closer to uncorrelated, the model no longer overestimates information from nearly-duplicate points.
#   As a result, the **Hessian becomes smaller**, the **Laplace approximation becomes broader**, and uncertainty estimates become more realistic.
#
#
# ## Model Considerations and Simplifications
#
# To compute $r_\Delta$, we need both
# $y_{n+1}^{\text{pred}}$ and $y_n^{\text{pred}}$.
# The previous version of the model uses **backward RK4** to infer $y_n^{\text{pred}}$ from $y_{n+1}^{\text{obs}}$. Although it seems that this is perfect for what we need, actually it is problematic.
#
# Indeed:
#
# $$
# y_n^{\text{pred}} = f_z^{-1}(y_{n+1}^{\text{obs}})
# $$
#
# uses **noisy future data**, so
#
# $$
# \text{noise}(y_n^{\text{pred}}) \sim J^{-1} \varepsilon_{n+1},
# $$
#
# which is **correlated** with $y_{n+1}^{\text{pred}}$. Thus we would need to derive a more complex $\Sigma_{\Delta}$, accounting for backward propagation of noise.
#
# Instead, if we try to avoid this issue by using the **true observation** at time $n$ rather than a backward model step:
#
# $$
# r_\Delta = (y_{n+1}^{\text{pred}} - y_n^{\text{obs}}) - (y_{n+1}^{\text{obs}} - y_n^{\text{obs}})
# = y_{n+1}^{\text{pred}} - y_{n+1}^{\text{obs}}.
# $$
#
# we end up with the old residual.
#
# Therefore, we need to update the model to:
#
# 1. Use **only forward prediction**
# 2. Consider **two steps at a time**
#
# ## 3 Assemble the increment residual
#
# Substitute the linearised forms and observed values:
#
# $$
# \begin{aligned}
# r^\text{obs}_\Delta
#   &\;=\;
#      (y_{n+1}^{\text{pred}}\!-\!y_{n }^{\text{pred}})
#      \;-\;
#      (y_{n+1}^{\text{obs}}\!-\!y_{n }^{\text{obs}}) \\[4pt]
#   &\approx
#      \bigl(y_{n+1}-y_n\bigr)
#      + \bigl(J_n\,\varepsilon_n - J_{n-1}\,\varepsilon_{n-1}\bigr) \\[-2pt]
#   &\quad
#      -\bigl(y_{n+1}-y_n\bigr)
#      - \bigl(\varepsilon_{n+1}-\varepsilon_{n}\bigr) \\[6pt]
#   &= r_\Delta + (J_n + I)\,\varepsilon_n \;-\; J_{n-1}\,\varepsilon_{n-1} \;-\; \varepsilon_{n+1}.
# \end{aligned}
# $$
#
# All three noise terms
# $\varepsilon_{n-1},\varepsilon_n,\varepsilon_{n+1}$
# are **independent** and have covariance $\Sigma_x$.
#
# Therefore the variance of $r_\Delta$ is:
# $$
# \operatorname{Var}[r_\Delta]
#    = (J_n + I)\,\Sigma_x\,(J_n + I)^{\!\top}
#      \;+\;
#      J_{n-1}\,\Sigma_x\,J_{\,n-1}^{\!\top}
#      \;+\;
#      \Sigma_x.
# $$
#
#
# And assuming we have  a **slowly-varying Jacobian** (constant J)
#    If $J_{n-1}\approx J_n \approx J$,
#
#    $$
#    \boxed{
#    \operatorname{Var}[r_\Delta]
#      \approx (J+I)\Sigma_x(J+I)^{\!\top} + J\Sigma_x J^{\!\top} + \Sigma_x.
#    }
#    $$
#
#
#
# ## 5 Interpretation
#
# * The first term $(J_n + I)\Sigma_x(J_n + I)^\top$ is the propagated
#   noise from the current measurement $\varepsilon_n$.
# * The second term $J_{n-1}\Sigma_xJ_{n-1}^\top$ is propagated noise from the
#   **previous** measurement $\varepsilon_{n-1}$.
# * The last term $\Sigma_x$ is the direct sensor noise at $n+1$.
# ---
#
# This derivation shows precisely how the minus sign and the independence of sensor noise translate into the residual covariance needed for a sound Gaussian likelihood on first-differences.
#
#

# %% [markdown]
# ## Covariance Between Residuals
#
# To compute the loss based on the **first-difference residual**
#
# $$
# r_{\Delta,n} = \Delta y^{\text{pred}} - \Delta y^{\text{obs}},
# $$
#
# we require two predicted values: $y_{n+1}^{\text{pred}}$ and $y_n^{\text{pred}}$.
#
# We consider different implementation strategies and evaluate how they affect the correlation between residuals, which in turn affects the **validity of assuming independence in the likelihood**.
#
# ---
#
# ### 1. **Using a Cached Previous Prediction**
#
# In this setup, residuals are computed consecutively:
#
# $$
# r_{\Delta,n} = (J_n + I)\varepsilon_n - J_{n-1}\varepsilon_{n-1} - \varepsilon_{n+1}
# $$
#
# $$
# r_{\Delta,n+1} = (J_{n+1} + I)\varepsilon_{n+1} - J_n \varepsilon_n - \varepsilon_{n+2}
# $$
#
# Compute the covariance:
#
# $$
# \begin{aligned}
# \operatorname{Cov}(r_{\Delta,n}, r_{\Delta,n+1})
# &= \mathbb{E}[r_{\Delta,n} r_{\Delta,n+1}^\top] \\
# &= \underbrace{\mathbb{E}[(J_n + I)\varepsilon_n \cdot (-J_n \varepsilon_n)^\top]}_{-(J_n + I)\Sigma_x J_n^\top}
# + \underbrace{\mathbb{E}[-\varepsilon_{n+1} \cdot (J_{n+1} + I)\varepsilon_{n+1}^\top]}_{-\Sigma_x (J_{n+1} + I)^\top}
# \end{aligned}
# $$
#
# Hence:
#
# $$
# \boxed{
# \operatorname{Cov}(r_{\Delta,n}, r_{\Delta,n+1}) =
# -(J_n + I)\Sigma_x J_n^\top - \Sigma_x (J_{n+1} + I)^\top
# }
# $$
#
# **Conclusion:** this strategy **induces residual correlation**, violating independence assumptions. The overlap between $r_{\Delta,n}$ and $r_{\Delta,n+1}$ (due to shared predictions) causes this.
#
# ---
#
# ### 2. **Partitioning Data into Disjoint Pairs**
#
# We define residuals over non-overlapping windows, e.g., predictions on $(n, n+1)$ and predictions on $(n+2, n+3)$:
#
# $$
# r_{\Delta,n} = (J + I)\varepsilon_n - J\varepsilon_{n-1} - \varepsilon_{n+1}
# $$
#
# $$
# r_{\Delta,n+2} = (J + I)\varepsilon_{n+2} - J \varepsilon_{n+1} - \varepsilon_{n+3}
# $$
#
# Here, both residuals share **only one common noise term**, $\varepsilon_{n+1}$, which appears as:
#
# * $-\varepsilon_{n+1}$ in $r_{\Delta,n}$
# * $-J \varepsilon_{n+1}$ in $r_{\Delta,n+2}$
#
# So:
#
# $$
# \operatorname{Cov}(r_{\Delta,n}, r_{\Delta,n+2}) = \mathbb{E}[-\varepsilon_{n+1} \cdot (-J \varepsilon_{n+1})^\top] = J \Sigma_x
# $$
#
# $$
# \boxed{
# \operatorname{Cov}(r_{\Delta,n}, r_{\Delta,n+2}) = J \Sigma_x
# }
# $$
#
# This is **non-zero**, so residuals still exhibit dependence — though they are more weakly correlated than in the cached case.
#
# ---
#
# ### 3. **Partitioning into Independent Triplets**
#
# Suppose we form non-overlapping triplets, e.g., $(x_0,x_1,x_2), (x_3,x_4,x_5),\dots$
#
# In this case, **each residual $r_{\Delta,k}$** depends only on:
#
# $$
# \varepsilon_{3k-1},\quad \varepsilon_{3k},\quad \varepsilon_{3k+1}
# $$
#
# and
#
# $$
# r_{\Delta,k+1} \text{ depends on } \varepsilon_{3k+2},\quad \varepsilon_{3k+3},\quad \varepsilon_{3k+4}
# $$
#
# So all noise terms are **disjoint** between residuals. Therefore:
#
# $$
# \boxed{
# \operatorname{Cov}(r_{\Delta,k}, r_{\Delta,k+1}) = 0
# }
# $$
#
# Residuals are **truly uncorrelated**, so the **likelihood decomposition into i.i.d. terms is valid**.
#
# ---
#
# ## In summary
#
# | Strategy                      | Residual Independence  | Data Utilization |
# | ----------------------------- | --------------------- | ---------------- |
# | **Cache previous prediction** | ❌ High correlation   | ✅ Full data      |
# | **Disjoint pairs**            | ⚠️ Some correlation   | ⚠️ 50% data      |
# | **Independent triplets**      | ✅ Fully independent  | ⚠️ 33% data      |
#
# For mathematical rigor we start by implementing the model that uses independent triplets. Then we can make some compromises to utilise the available data more fully.
#

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
        total += ((proposed_value - mu) / sig) ** 2 / 2
    return total


def _parse_data_noise_to_sigma(data_noise: Union[float, Iterable, torch.Tensor]) -> torch.Tensor:
    """Parse data_noise and return the inverse covariance matrix Sigma_x_inv."""
    if isinstance(data_noise, float):
        a = data_noise
        b = data_noise
        return torch.diag(torch.tensor([a, b], dtype=torch.float32))
    elif isinstance(data_noise, torch.Tensor):
        if data_noise.shape != (2, 2):
            raise ValueError("If data_noise is a tensor, it must be 2x2.")
        return data_noise
    elif isinstance(data_noise, Iterable):
        data_noise = list(data_noise)
        if len(data_noise) != 2:
            raise ValueError("If data_noise is iterable, it must be of length 2.")
        a = data_noise[0]
        b = data_noise[1]
        return torch.diag(torch.tensor([a, b], dtype=torch.float32))
    else:
        raise TypeError("data_noise must be float, 2-tensor, or iterable of length 2.")


def likelihood_loss(preds, y_n, y_np1, Sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute −log likelihood with Mahalanobis norm using per-variable covariance.
    preds: tuple of (i_n, v_n, i_np1, v_np1)
    y_n, y_np1: true values at time steps n and n+1
    Sigma_x_inv: 2x2 inverse covariance matrix for [i, v]
    """
    i_n, v_n, i_np1, v_np1 = preds
    i0, v0 = y_n[:, 0:1], y_n[:, 1:2]
    i1, v1 = y_np1[:, 0:1], y_np1[:, 1:2]

    # Residuals: shape [N, 2]
    res_0 = torch.cat([i_n - i0, v_n - v0], dim=1)
    res_1 = torch.cat([i_np1 - i1, v_np1 - v1], dim=1)

    # Stack both sets of residuals: shape [2N, 2]
    residuals = torch.cat([res_0, res_1], dim=0)

    # Cholensky approach
    L = torch.linalg.cholesky(Sigma)                    # (2,2), lower-tri
    z = torch.linalg.solve_triangular(L, residuals.T, upper=False).T
    return 0.5 * z.pow(2).sum()


def likelihood_loss_triplets(
    preds: torch.Tensor, targets: torch.Tensor, Sigma: torch.Tensor  # (N,4)  # (N,4)  # (2,2)
) -> torch.Tensor:
    """
    Negative log-likelihood for Δ-residuals under
    r_Δ ∼ 𝒩(0, Σ),    Σ = Sigma (2×2).
    """
    # --- split columns ------------------------------------------------
    i_np1_pred, v_np1_pred, i_n_pred, v_n_pred = preds.T  # (N,)
    i_np1_true, v_np1_true, i_n_true, v_n_true = targets.T  # (N,)

    # --- build increments --------------------------------------------
    delta_pred = torch.stack([i_np1_pred - i_n_pred, v_np1_pred - v_n_pred], dim=1)  # (N,2)
    delta_obs = torch.stack([i_np1_true - i_n_true, v_np1_true - v_n_true], dim=1)  # (N,2)

    residuals = delta_pred - delta_obs  # (N,2)

    # --- Mahalanobis term via whitening ------------------------------
    L = torch.linalg.cholesky(Sigma)  # (2,2)
    z = torch.linalg.solve_triangular(L, residuals.T, upper=False).T  # (N,2)
    nll = 0.5 * z.pow(2).sum()  # scalar
    return nll


# define the loss function for MAP estimation that combines the L2 loss and the log-normal prior.
def make_map_loss(
    nominal: Parameters, sigma: Parameters, residual_covariance: Union[float, Iterable, torch.Tensor] = 1.0
) -> Callable:

    residual_covariance = _parse_data_noise_to_sigma(residual_covariance)
    def _loss(model, preds, targets):
        ll = likelihood_loss_triplets(preds, targets, residual_covariance)
        prior = log_normal_prior(model.logparams, nominal, sigma) 
        return ll + prior

    return _loss

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
# ---------------------------------------------------------------------
#  Trainer for the triplet-based BuckParamEstimator
# ---------------------------------------------------------------------
from typing import Callable, Dict, List, Any
from functools import partial


class TrainerTriplets:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optim_cfg: "AdamOptTrainingConfigs",
        device: str = "cpu",
    ):
        self.model = model.to(device)
 # expects (preds, targets)
        self.optim_cfg = optim_cfg
        self.device = device
        self.loss_fn = loss_fn
        # history for plotting / CSV export
        self.history: Dict[str, List[Any]] = {"loss": [], "params": [], "lr": []}

    # -----------------------------------------------------------------
    def _record(self, loss_val: float):
        est = self.model.get_estimates()
        self.history["loss"].append(loss_val)
        self.history["params"].append(est)
        # LR from the first param-group (Adam & LBFGS both expose it)
        self.history["lr"].append(self.opt.param_groups[0]["lr"])

    # -----------------------------------------------------------------
    def fit(self, X: torch.Tensor, epochs_adam: int = 20_000, epochs_lbfgs: int = 500):

        X = X.to(self.device)

        # ------------- Adam phase ------------------------------------
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.optim_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=self.optim_cfg.lr_reduction_factor,
            patience=self.optim_cfg.patience,
        )

        best_loss = float("inf")

        for it in range(epochs_adam):
            self.opt.zero_grad()
            preds, targets = self.model(X)  # <- new API
            loss = self.loss_fn(self.model, preds, targets)
            loss.backward()
            self.opt.step()
            scheduler.step(loss.item())

            if (it % 1000) == 0:
                self._record(loss.item())
                best_loss = min(best_loss, loss.item())

                est = self.model.get_estimates()
                grad_norm = (
                    torch.cat(
                        [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]
                    )
                    .norm()
                    .item()
                )

                print(
                    f"[Adam {it:>6}] "
                    f"loss={loss.item():.3e}, grad‖={grad_norm:.3e}, "
                    f"L={est.L:.2e}, C={est.C:.2e}, "
                    f"Rload1={est.Rload1:.2e}, Rload2={est.Rload2:.2e}, "
                    f"Rload3={est.Rload3:.2e}"
                )

        print("Adam finished.  Best loss:", best_loss)

        # ------------- LBFGS phase -----------------------------------
        print("Starting LBFGS optimisation …")
        self.opt = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=epochs_lbfgs,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-8,
        )

        def closure():
            self.opt.zero_grad()
            preds, targets = self.model(X)
            loss = self.loss_fn(self.model, preds, targets)
            loss.backward()
            return loss

        self.opt.step(closure)
        final_loss = closure().item()
        self._record(final_loss)
        print("LBFGS finished.  Final loss:", final_loss)
        # print the final parameter estimations
        print("Final parameter estimates:")
        est = self.model.get_estimates()
        print(
            f"L={est.L:.2e}, C={est.C:.2e}, "
            f"Rload1={est.Rload1:.2e}, Rload2={est.Rload2:.2e}, "
            f"Rload3={est.Rload3:.2e},"
        )

        # ------------- save history ----------------------------------
        run = TrainingRun.from_histories(
            loss_history=self.history["loss"],
            param_history=self.history["params"],
        )
        out_dir = self.optim_cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_name = (
            self.optim_cfg.savename
            if str(self.optim_cfg.savename).endswith(".csv")
            else f"{self.optim_cfg.savename}.csv"
        )
        run.save_to_csv(out_dir / csv_name)

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

# %% [markdown]
# Now let's estimate the Frobenius norm by using the autograd function included in pytorch.
#
# The best solution would be to **pretrain the model to be close to the correct physical quantities of the parameters**. But for simplicity we can use the nominal values.
#
# Moreover, the noise on i and the noise on v may be very different in magnitude. However, again for simplicity we calculate a single Frobenius norm for both.

# %%
from torch.autograd.functional import jacobian

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


def estimate_avg_Jacobian(model, X, y, max_samples=10)-> torch.Tensor:
    model.eval()
    X = X.detach()
    y = y.detach()

    jacobians = []

    for x_n, y_n in zip(X[:max_samples], y[:max_samples]):

        x_n = x_n.detach()
        x_n.requires_grad_(False)

        # Extract D and dt from x_n (they’re fixed)
        D = x_n[2].unsqueeze(0)
        dt = x_n[3].unsqueeze(0)

        # Define function of ONLY the noisy inputs: i, v
        def f_noisy_inputs(i_v):
            x_full = torch.cat([i_v, D, dt], dim=0).unsqueeze(0)
            y_input = y_n.unsqueeze(0)
            i_pred, v_pred = model(x_full, y_input)[2:]
            return torch.cat([i_pred, v_pred], dim=1).squeeze()  # shape (2,)

        i_v_input = x_n[:2].clone().detach().requires_grad_(True)
        J = jacobian(f_noisy_inputs, i_v_input)  # shape (2, 2)
        jacobians.append(J)
    return torch.stack(jacobians).mean(0)  # average over all jacobians


jacobians = {}
# Loop through all groups and estimate the Frobenius norm for each group

for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    # Load the data from the hdf5 file
    io = LoaderH5(db_dir, h5filename)
    io.load(group_name)

    X_t, y_t, model = load_data_to_model(io.M, initial_guess_params=NOMINAL)

    print(f"Estimating Jacobian for group {group_name}...")
    jac = estimate_avg_Jacobian(model, X_t, y_t, max_samples=300)
    jacobians[group_name] = jac
    print(f"Jacobain for group {group_name} ({jac.shape}): {jac}")

# average the Frobenius norms across all groups
J_av = torch.stack(list(jacobians.values())).mean(0)
print(f"Average Frobenius norm across all groups: {jacobian}")

# %%
# what is the error in the Frobenius norm estimation, with respect to the true parameter values?
for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    # Load the data from the hdf5 file
    io = LoaderH5(db_dir, h5filename)
    io.load(group_name)

    X_t, y_t, model_tr = load_data_to_model(io.M, initial_guess_params=TRUE_PARAMS)

    print(f"Estimating Jacobian for group {group_name}...")
    jac = estimate_avg_Jacobian(model_tr, X_t, y_t, max_samples=300)
    jacobians[group_name] = jac
    print(f"Jacobain for group {group_name} ({jac.shape}): {jac}")

# %%
def estimate_avg_frobenius_norm(model, X, y, max_samples=10):
    model.eval()
    X = X.detach()
    y = y.detach()

    norms_squared = []

    for x_n, y_n in zip(X[:max_samples], y[:max_samples]):

        x_n = x_n.detach()
        x_n.requires_grad_(False)

        # Extract D and dt from x_n (they’re fixed)
        D = x_n[2].unsqueeze(0)
        dt = x_n[3].unsqueeze(0)

        # Define function of ONLY the noisy inputs: i, v
        def f_noisy_inputs(i_v):
            x_full = torch.cat([i_v, D, dt], dim=0).unsqueeze(0)
            y_input = y_n.unsqueeze(0)
            i_pred, v_pred = model(x_full, y_input)[2:]
            return torch.cat([i_pred, v_pred], dim=1).squeeze()  # shape (2,)

        i_v_input = x_n[:2].clone().detach().requires_grad_(True)
        J = jacobian(f_noisy_inputs, i_v_input)  # shape (2, 2)
        frob_norm_sq = torch.norm(J, p="fro") ** 2
        norms_squared.append(frob_norm_sq.item())

    return sum(norms_squared) / len(norms_squared)


# compare with the Frobenius norm used previously
frob_sq_direct = torch.norm(J_av, p="fro").pow(2)
frob_sq_avg = estimate_avg_frobenius_norm(model, X_t, y_t, max_samples=300)

print(f"Frobenius norm squared (direct): {frob_sq_direct:.3e}")
print(f"Frobenius norm squared (average): {frob_sq_avg:.3e}")

# %% [markdown]
# We can see that the values of the Frobenius norm with the true parameters are quite close to the one obtained with the nominal parameters!
#
# Now we can rescale the noise powers using the formula:
#
#    $$
#    \boxed{
#    \operatorname{Var}[r_\Delta]
#      \approx (J+I)\Sigma_x(J+I)^{\!\top} + J\Sigma_x J^{\!\top} + \Sigma_x.
#    }
#    $$

# %%
# rescale the noise powers
Sigma_x_ADC = torch.tensor([[noise_power_ADC_i, 0], [0, noise_power_ADC_v]], dtype=torch.float32)
Sigma_x_5 = torch.tensor([[noise_power_5_i, 0], [0, noise_power_5_v]], dtype=torch.float32)
Sigma_x_10 = torch.tensor([[noise_power_10_i, 0], [0, noise_power_10_v]], dtype=torch.float32)

# transform noise power to the covariance matrix of the residuals
def transform_noise_to_residual_covariance(noise_power: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Transform noise power to the covariance matrix of the residuals."""
    # J is the Jacobian, noise_power is a 2x2 diagonal matrix
    return (J + torch.eye(2)) @ noise_power @ (J + torch.eye(2)).T + J @ noise_power @ J.T + noise_power

# Propagate the noise through the Jacobian to get the rescaled noise power
Sigma_tot_ADC = transform_noise_to_residual_covariance(Sigma_x_ADC, J_av)
Sigma_tot_5 = transform_noise_to_residual_covariance(Sigma_x_5, J_av)
Sigma_tot_10 = transform_noise_to_residual_covariance(Sigma_x_10, J_av)

print(f"Sigma_tot_ADC:\n{Sigma_tot_ADC}")
print(f"Sigma_tot_5:\n{Sigma_tot_5}")
print(f"Sigma_tot_10:\n{Sigma_tot_10}")


# %%
from pinn_buck.model.model_param_estimator import BuckParamEstimatorTriplets

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
    model = BuckParamEstimatorTriplets(lb, ub, s1, s2, s3, initial_guess_params).to(device)
    return X_t, y_t, model

noise_power_dict = {
    0: 1e-9,  # ideal
    1: Sigma_tot_ADC,  # ADC error
    3: Sigma_tot_5,  # 5 noise
    4: Sigma_tot_10,  # 10 noise
}


noisy_measurements = {}
trained_models = {}
inverse = False

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

    trainer = TrainerTriplets(
        model=model,
        loss_fn=make_map_loss(
            **prior_info,
            residual_covariance=noise_power_dict[group_number],  # use the noise power for the group
        ),
        optim_cfg=run_configs,
        device=device,
    )

    trainer.fit(
        X=X,
        epochs_adam=run_configs.epochs,
        epochs_lbfgs=run_configs.epochs_lbfgs,
    )
    inverse = True  # inverse is False only for the ideal case, so we set it to True for the rest of the groups
    trained_models[group_name] = trainer.model
    print("\n \n \n")
print("done")

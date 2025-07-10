import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from pinn_buck.config import Parameters

# True values
TRUE = Parameters(
    L=7.25e-4,
    RL=0.314,
    C=1.645e-4,
    RC=0.201,
    Rdson=0.221,
    Rload1=3.1,
    Rload2=10.2,
    Rload3=6.1,
    Vin=48.0,
    VF=1.0,
)

# Nominals and linear-space relative tolerances
NOMINAL = Parameters(
    L=6.8e-4,
    RL=0.3,
    C=1.5e-4,
    RC=0.25,
    Rdson=0.2,
    Rload1=3.3,
    Rload2=10.0,
    Rload3=6.8,
    Vin=48.0,
    VF=0.9,
)

REL_TOL = Parameters(
    L=0.20,
    RL=0.33,
    C=0.20,
    RC=0.40,
    Rdson=0.10,
    Rload1=0.05,
    Rload2=0.05,
    Rload3=0.05,
    Vin=0.02,
    VF=0.11,
)

# Plotting
param_names = Parameters._fields
ncols = 2
nrows = int(np.ceil(len(param_names) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))
axes = axes.flatten()

for i, name in enumerate(param_names):
    nominal = getattr(NOMINAL, name)
    rel_tol = getattr(REL_TOL, name)
    true_val = getattr(TRUE, name)

    # log-normal parameters
    sigma = np.log(1 + rel_tol)
    mu = np.log(nominal)

    dist = lognorm(s=sigma, scale=np.exp(mu))

    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
    pdf = dist.pdf(x)

    ax = axes[i]
    ax.plot(x, pdf, label=f"{name} prior")
    ax.axvline(true_val, color="red", linestyle="--", label="TRUE")
    ax.set_title(name)
    ax.set_yticks([])
    ax.legend()

fig.suptitle("Log-Normal Priors (linear space) with TRUE values", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

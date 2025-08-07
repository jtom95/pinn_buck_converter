from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import h5py
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Parameters container & scaling helpers
# ---------------------------------------------------------------------
# add the root directory of the project to the path
sys.path.append(str(Path(__file__).parent.parent))

from pinn_buck.config import Parameters, TrainingRun
from pinn_buck.config import TRUE as TRUE_PARAMS, INITIAL_GUESS as INITIAL_GUESS_PARAMS
from pinn_buck.config import _SCALE

from pinn_buck.noise import add_noise_to_Measurement

from pinn_buck.parameter_transformation import make_log_param, reverse_log_param
from pinn_buck.io import LoaderH5


def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# PINN model
# ---------------------------------------------------------------------


# Main script
set_seed(123)
device = "cpu"

# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"

io = LoaderH5(db_dir, h5filename)
io.load("ideal")

ideal_meas = io.M

lr = 1e-3
epochs_1 = 30_000
epochs_2 = 20_000
patience = 5000
device = "cpu"  # or "cuda" if you have a GPU
lr_reduction_factor = 0.5


# NOISE PARAMETERS
VFS = 30  # Full scale voltage
IFS = 10  # Full scale current
number_of_noise_samples = 10


noise_levels = [
    1, 
    3, 
    5, 
    10
]
out_dir = Path(__file__).parent.parent / "RESULTS" / "Adam_Opt" / "noisy_runs" / "DATA"
out_dir.mkdir(parents=True, exist_ok=True)


for idx, noise in enumerate(noise_levels):
    print(f"\n{'-'*50}")
    print(f"{idx}) Generating with noise level: {noise}")

    for idx in range(number_of_noise_samples):
        print(f"Sample {idx + 1} of {number_of_noise_samples}")

        noisy_meas = add_noise_to_Measurement(ideal_meas, noise_level=noise, V_FS=VFS, I_FS=IFS)

        # save noisy measurements
        noisy_meas.save_to_numpyzip(
            out_dir / f"noise_{noise}LSB_sample_{idx + 1}.npz",
        )
## ---------------------------------------------------------------------
## L-BFGS optimization
## ---------------------------------------------------------------------

# lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=50000, tolerance_grad=1e-9)
# def closure():
#     lbfgs.zero_grad()
#     l = compute_loss(model(X_t), x0, y_t)
#     l.backward()
#     return l
# print("Starting L-BFGS …")
# lbfgs.step(closure)

# # Report
# est = model.get_estimates()
# def rel_err(est, ref): return abs(est / ref - 1) * 100
# for name in Parameters._fields:
#     val, nom = getattr(est, name), getattr(parameters_nominal, name)
#     print(f"{name:>7s}: {val:.3e}  (error = {rel_err(val, nom):.2f} %)")
#

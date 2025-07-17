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


def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# ---------------------------------------------------------------------


def train_from_measurement_file(
    meas: Measurement, 
    savename: str = "saved_run",
    db_dir: Path = ".",
    lr: float = 1e-3,
    epochs1: int = 30_000,
    epochs2: int = 30_000,
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
    model = BuckParamEstimator(lb, ub, s1, s2, s3, INITIAL_GUESS_PARAMS).to(device)

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

    for it in range(epochs1 + epochs2):
        optimizer.zero_grad()
        pred = model(X_t, y_t)
        # loss = compute_loss(pred, x0, y_t) if it < epochs1 else compute_L1_loss(pred, x0, y_t)
        
        loss = l2_loss(pred, x0, y_t)
        loss.backward()
        
        # # when we change loss let's reset the lr
        if it == epochs1:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * 0.1 # reset the learning rate  
                
        
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

    # # generate the output directory if it doesn't exist
    # db_dir.mkdir(parents=True, exist_ok=True)

    # # if savename doesn't end with .csv, add it
    # if not savename.endswith(".csv"):
    #     savename += ".csv"

    # training_run.save_to_csv(db_dir / savename)
    # print("Concluded ADAM training.")

# Main script
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
epochs_1 = 30_000
epochs_2 = 20_000
patience = 5000
device = "cpu"  # or "cuda" if you have a GPU
lr_reduction_factor = 0.5


out_dir = Path(__file__).parent.parent / "RESULTS" / "Bayesian" / "Adam"

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
    
    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")
    
    # Train the model on the noisy measurement
    train_from_measurement_file(
        io.M,
        db_dir=out_dir,
        savename=f"noisy_run_{group_name}.csv",
        lr=lr,
        patience=patience,
        lr_reduction_factor=lr_reduction_factor,
        epochs1=epochs_1,
        epochs2=epochs_2,
        device=device,
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

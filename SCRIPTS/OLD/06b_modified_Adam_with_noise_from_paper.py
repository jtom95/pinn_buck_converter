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


from pinn_buck.parameter_transformation import make_log_param, reverse_log_param
from pinn_buck.io import LoaderH5


def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# PINN model
# ---------------------------------------------------------------------


class BuckParamEstimator(nn.Module):
    """Physics‑informed NN for parameter estimation in a buck converter."""

    def __init__(
        self,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        split1: int,
        split2: int,
        split3: int,
        param_init: Parameters,
    ) -> None:
        super().__init__()
        self.lb = torch.as_tensor(lower_bound, dtype=torch.float32)
        self.ub = torch.as_tensor(upper_bound, dtype=torch.float32)
        self.s1, self.s2, self.s3 = split1, split2, split3

        # Trainable log‑parameters
        log_params = make_log_param(param_init)
        self.log_L = nn.Parameter(log_params.L, requires_grad=True)
        self.log_RL = nn.Parameter(log_params.RL, requires_grad=True)
        self.log_C = nn.Parameter(log_params.C, requires_grad=True)
        self.log_RC = nn.Parameter(log_params.RC, requires_grad=True)
        self.log_Rdson = nn.Parameter(log_params.Rdson, requires_grad=True)
        self.log_Rload1 = nn.Parameter(log_params.Rload1, requires_grad=True)
        self.log_Rload2 = nn.Parameter(log_params.Rload2, requires_grad=True)
        self.log_Rload3 = nn.Parameter(log_params.Rload3, requires_grad=True)
        self.log_Vin = nn.Parameter(log_params.Vin, requires_grad=True)
        self.log_VF = nn.Parameter(log_params.VF, requires_grad=True)

    # ----------------------------- helpers -----------------------------
    @property
    def logparams(self) -> Parameters:
        """Return current log‑space parameters."""
        return Parameters(
            L=self.log_L,
            RL=self.log_RL,
            C=self.log_C,
            RC=self.log_RC,
            Rdson=self.log_Rdson,
            Rload1=self.log_Rload1,
            Rload2=self.log_Rload2,
            Rload3=self.log_Rload3,
            Vin=self.log_Vin,
            VF=self.log_VF,
        )

    def _scale(self, x: torch.Tensor):
        return 2 * (x - self.lb) / (self.ub - self.lb) - 1

    def _physical(self) -> Parameters:
        """Return current parameters in physical units (inverse scaling)."""
        return reverse_log_param(self.logparams)

    def get_estimates(self) -> Parameters:
        """Return current parameters in physical units."""
        params = self._physical()
        return Parameters(
            L=params.L.item(),
            RL=params.RL.item(),
            C=params.C.item(),
            RC=params.RC.item(),
            Rdson=params.Rdson.item(),
            Rload1=params.Rload1.item(),
            Rload2=params.Rload2.item(),
            Rload3=params.Rload3.item(),
            Vin=params.Vin.item(),
            VF=params.VF.item(),
        )

    # ---------------------- physics right‑hand sides -------------------
    @staticmethod
    def _di(i_k, v_k, S, p: Parameters):
        return -((S * p.Rdson + p.RL) * i_k + v_k - S * p.Vin + (1 - S) * p.VF) / p.L

    @staticmethod
    def _dv(i_k, v_k, S, p: Parameters, rload, di):
        return (p.C * p.RC * rload * di + rload * i_k - v_k) / (p.C * (p.RC + rload))

    # ------------------------------- forward --------------------------
    @staticmethod
    def _rk4_step(i, v, D, dt, p: Parameters, rload, sign=+1):

        dh = dt * sign

        def f(i_, v_):
            di = -((D * p.Rdson + p.RL) * i_ + v_ - D * p.Vin + (1 - D) * p.VF) / p.L
            dv = (p.C * p.RC * rload * di + rload * i_ - v_) / (p.C * (p.RC + rload))
            return di, dv

        k1_i, k1_v = f(i, v)
        k2_i, k2_v = f(i + 0.5 * dh * k1_i, v + 0.5 * dh * k1_v)
        k3_i, k3_v = f(i + 0.5 * dh * k2_i, v + 0.5 * dh * k2_v)
        k4_i, k4_v = f(i + dh * k3_i, v + dh * k3_v)

        i_new = i + dh / 6.0 * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        v_new = v + dh / 6.0 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return i_new, v_new

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        i_n, v_n = X[:, 0:1], X[:, 1:2]
        S, dt = X[:, 2:3], X[:, 3:4]

        i_np1, v_np1 = y[:, 0:1], y[:, 1:2]

        p = self._physical()
        rload = torch.cat(
            [
                torch.ones((self.s1, 1), device=X.device) * p.Rload1,
                torch.ones((self.s2, 1), device=X.device) * p.Rload2,
                torch.ones((self.s3, 1), device=X.device) * p.Rload3,
            ],
            dim=0,
        )

        # forward prediction using RK4
        i_np1_pred, v_np1_pred = self._rk4_step(
            i_n, v_n, S, dt, p, rload, sign=+1
        )

        # backward prediction using RK4
        i_n_pred, v_n_pred = self._rk4_step(
            i_np1, v_np1, S, dt, p, rload, sign=-1
        )
        return i_n_pred, v_n_pred, i_np1_pred, v_np1_pred


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------


def compute_loss(preds: torch.Tensor, y_n: torch.Tensor, y_np1: torch.Tensor) -> torch.Tensor:
    i_n, v_n, i_np1, v_np1 = preds
    i0, v0 = y_n[:, :1], y_n[:, 1:]
    i1, v1 = y_np1[:, :1], y_np1[:, 1:]
    return (
        (i_n - i0).pow(2).sum()
        + (v_n - v0).pow(2).sum()
        + (i_np1 - i1).pow(2).sum()
        + (v_np1 - v1).pow(2).sum()
    )


def compute_L1_loss(preds: torch.Tensor, y_n: torch.Tensor, y_np1: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 regularization loss for the parameters.
    This is used to encourage sparsity in the parameter estimates.
    """
    i_n, v_n, i_np1, v_np1 = preds
    i0, v0 = y_n[:, :1], y_n[:, 1:]
    i1, v1 = y_np1[:, :1], y_np1[:, 1:]
    
    # L1 regularization on the predictions
    return (
        torch.abs(i_n - i0).sum()
        + torch.abs(v_n - v0).sum()
        + torch.abs(i_np1 - i1).sum()
        + torch.abs(v_np1 - v1).sum()
    )

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------

from pinn_buck.io import Measurement
from pinn_buck.noise import add_noise_to_Measurement


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

    # Optimisation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_reduction_factor, patience=patience
    )

    for it in range(epochs1 + epochs2):
        optimizer.zero_grad()
        pred = model(X_t, y_t)
        loss = compute_loss(pred, x0, y_t) if it < epochs1 else compute_L1_loss(pred, x0, y_t)
        # the L1 loss is more robust to outliers, so we use it after the first phase of training
        loss.backward()
        
        # when we change loss let's reset the lr
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

    # generate the output directory if it doesn't exist
    db_dir.mkdir(parents=True, exist_ok=True)

    # if savename doesn't end with .csv, add it
    if not savename.endswith(".csv"):
        savename += ".csv"

    training_run.save_to_csv(db_dir / savename)
    print("Concluded ADAM training.")

# Main script
set_seed(123)
device = "cpu"

# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"

GROUP_NUMBER_DICT = {
    0: "ideal",
    1: "ADC_error",
    2: "Sync Error",
    3: "5 noise",
    4: "10 noise",
    5: "ADC-Sync-5noise",
    6: "ADC-Sync-10noise",
}


lr = 1e-3
epochs_1 = 30_000
epochs_2 = 30_000
patience = 5000
device = "cpu"  # or "cuda" if you have a GPU
lr_reduction_factor = 0.5



out_dir = Path(__file__).parent.parent / "RESULTS" / "removed_nn" / "noisy_runs_from_paper_finetuning_L1"

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
#     val, nom = getattr(est, name), getattr(parameters_TRUE, name)
#     print(f"{name:>7s}: {val:.3e}  (error = {rel_err(val, nom):.2f} %)")
#

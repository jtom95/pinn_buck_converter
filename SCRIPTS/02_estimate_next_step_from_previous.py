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
from pinn_buck.config import NOMINAL as NOMINAL_PARAMS, INITIAL_GUESS as INITIAL_GUESS_PARAMS
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


class PINNBuck(nn.Module):
    """Physics‑informed NN for parameter estimation in a buck converter."""

    def __init__(
        self,
        layers: List[int],
        steps_irk: int,
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
        self.q = max(int(steps_irk), 1)
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

        # FC network backbone
        net: List[nn.Module] = []
        for m, n in zip(layers[:-1], layers[1:]):
            net.append(nn.Linear(m, n))
            if n != layers[-1]:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

        # IRK constants
        bfile = f"Butcher_tableau/Butcher_IRK{self.q}.txt"
        alpha_beta_times = np.loadtxt(bfile, dtype=np.float32)
        pivot = self.q * self.q + self.q
        mats = alpha_beta_times[:pivot].reshape(self.q + 1, self.q)
        self.register_buffer("irk_alpha", torch.tensor(mats[:-1]))
        self.register_buffer("irk_beta", torch.tensor(mats[-1:]))

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
    def predict_next_state(
        i_n: torch.Tensor,
        v_n: torch.Tensor,
        D: torch.Tensor,
        dt: torch.Tensor,
        L: torch.Tensor,
        RL: torch.Tensor,
        C: torch.Tensor,
        RC: torch.Tensor,
        Rdson: torch.Tensor,
        Rload: torch.Tensor,
        Vin: torch.Tensor,
        Vf: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next state [i_np1, v_np1] using the RK4 method."""

        def f(i: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            di = -((D * Rdson + RL) * i + v - D * Vin + (1 - D) * Vf) / L
            dv = (C * RC * Rload * di + Rload * i - v) / (C * (RC + Rload))
            return di, dv

        # RK4 steps
        k1_i, k1_v = f(i_n, v_n)
        k2_i, k2_v = f(i_n + 0.5 * dt * k1_i, v_n + 0.5 * dt * k1_v)
        k3_i, k3_v = f(i_n + 0.5 * dt * k2_i, v_n + 0.5 * dt * k2_v)
        k4_i, k4_v = f(i_n + dt * k3_i, v_n + dt * k3_v)

        i_np1_pred = i_n + (dt / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        v_np1_pred = v_n + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return i_np1_pred, v_np1_pred
    @classmethod
    def predict_previous_state(
        cls, 
        i_np1: torch.Tensor, 
        v_np1: torch.Tensor, 
        D: torch.Tensor, 
        dt: torch.Tensor, 
        L: torch.Tensor, 
        RL: torch.Tensor, 
        C: torch.Tensor, 
        RC: torch.Tensor, 
        Rdson: torch.Tensor, 
        Rload: torch.Tensor, 
        Vin: torch.Tensor, 
        Vf: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Predict the previous state [i_n, v_n] using the RK4 method."""
        return cls.predict_next_state(
            i_n=i_np1,
            v_n=v_np1,
            D=D,
            dt=-dt,  # negative dt to reverse the direction
            L=L,
            RL=RL,
            C=C,
            RC=RC,
            Rdson=Rdson,
            Rload=Rload,
            Vin=Vin,
            Vf=Vf
        )


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

        i_np1_pred, v_np1_pred = self.predict_next_state(
            i_n, v_n, S, dt, p.L, p.RL, p.C, p.RC, p.Rdson, rload, p.Vin, p.VF
        )
        i_n_pred, v_n_pred = self.predict_previous_state(
            i_np1, v_np1, S, dt, p.L, p.RL, p.C, p.RC, p.Rdson, rload, p.Vin, p.VF
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


def set_seed(s):
    np.random.seed(s) 
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# ---------------------------------------------------------------------
# Main script
set_seed(123)
device = "cpu"

# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"


io = LoaderH5(db_dir, h5filename)
io.load("ideal")

# load the transient data as unified numpy arrays
X, y = io.data
s1, s2, s3 = list(map(lambda x: x-1, io.transient_lengths)) # subtract 1 since we use the previous time step as input
lb, ub = X.min(0), X.max(0)


X_t = torch.tensor(X, device=device)
y_t = torch.tensor(y, device=device)
x0 = X_t[:, :2]

# Model
q = 20
layers = [4, 50, 50, 50, 50, 50, q * 2]
model = PINNBuck(layers, q, lb, ub, s1, s2, s3, INITIAL_GUESS_PARAMS).to(device)

# params
lr = 1e-3
epochs = 50_000

history_loss = []
history_params: List[Parameters] = []


# Optimisation
adam = torch.optim.Adam(model.parameters(), lr=lr)
for it in range(epochs):
    adam.zero_grad()
    pred = model(X_t, y_t)
    loss = compute_loss(pred, x0, y_t)
    loss.backward()
    adam.step()
    if it % 1000 == 0:

        est = model.get_estimates()
        history_loss.append(loss.item())
        history_params.append(est)

        # print the parameter estimation
        est = model.get_estimates()
        print(f"Iteration {it}, loss {loss:4e},  Parameters (Adam):", 
                f"L={est.L:.3e}, RL={est.RL:.3e}, C={est.C:.3e}, "
                f"RC={est.RC:.3e}, Rdson={est.Rdson:.3e}, "
                f"Rload1={est.Rload1:.3e}, Rload2={est.Rload2:.3e}, "
                f"Rload3={est.Rload3:.3e}, Vin={est.Vin:.3f}, VF={est.VF:.3e}")


df = pd.DataFrame({
    "loss": history_loss,
    "L": [p.L for p in history_params],
    "RL": [p.RL for p in history_params],
    "C": [p.C for p in history_params],
    "RC": [p.RC for p in history_params],
    "Rdson": [p.Rdson for p in history_params],
    "Rload1": [p.Rload1 for p in history_params],
    "Rload2": [p.Rload2 for p in history_params],
    "Rload3": [p.Rload3 for p in history_params],
    "Vin": [p.Vin for p in history_params],
    "VF": [p.VF for p in history_params]
})


# # Save the history to a CSV file
# save_dir = Path(__file__).parent.parent / "RESULTS" / "original_structure" 
# history_file = save_dir / "rk4.csv"
# save_dir.mkdir(parents=True, exist_ok=True)
# df.to_csv(history_file, index=False)

print("Concluded ADAM training.")

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

"""
==================================================================================================
PyTorch re‑implementation of:
    S. Zhao, Y. Peng, Y. Zhang, and H. Wang,
    "Parameter Estimation of Power Electronic Converters with Physics‑informed Machine Learning",
    IEEE Trans. Power Electronics, 2022.

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import h5py
# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OtherParams:
    """Hyper‑parameters & nominal values (mirrors original otherParams)."""

    def __init__(self):
        # Optimisation
        self.adam_epochs: int = 45_000
        self.adam_lr: float = 1e-3
        self.lbfgs_epochs: int = 50_000

        # Nominal component values (Table I in the paper)
        self.nL = 7.25
        self.nRL = 3.14
        self.nC = 1.645
        self.nRC = 2.01
        self.nRdson = 2.21
        self.nRload1 = 3.1
        self.nRload2 = 10.2
        self.nRload3 = 6.1
        self.nVin = 4.8
        self.nVF = 1.0


# -----------------------------------------------------------------------------
# Physics‑informed neural network module
# -----------------------------------------------------------------------------


class PINNBuck(nn.Module):
    """Physics‑Informed Neural Network for buck converter parameter estimation."""

    def __init__(
        self,
        layers: List[int],
        steps_irk: int,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        split_idx1: int,
        split_idx2: int,
        split_idx3: int,
    ) -> None:
        super().__init__()

        self.lower_bound = torch.tensor(lower_bound, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.float32)
        self.steps_irk = max(steps_irk, 1)
        self.split_idx1 = split_idx1
        self.split_idx2 = split_idx2
        self.split_idx3 = split_idx3

        # ------------------------------------------------------------------
        # Trainable log‑parameters (initialised to TF defaults)
        # ------------------------------------------------------------------
        self.log_L = nn.Parameter(torch.log(torch.tensor([2.0], dtype=torch.float32)))
        self.log_RL = nn.Parameter(torch.log(torch.tensor([0.039], dtype=torch.float32)))
        self.log_C = nn.Parameter(torch.log(torch.tensor([0.412], dtype=torch.float32)))
        self.log_RC = nn.Parameter(torch.log(torch.tensor([1.59], dtype=torch.float32)))
        self.log_Rdson = nn.Parameter(torch.log(torch.tensor([1.22], dtype=torch.float32)))
        self.log_Rload1 = nn.Parameter(torch.log(torch.tensor([1.22], dtype=torch.float32)))
        self.log_Rload2 = nn.Parameter(torch.log(torch.tensor([1.22], dtype=torch.float32)))
        self.log_Rload3 = nn.Parameter(torch.log(torch.tensor([1.22], dtype=torch.float32)))
        self.log_vin = nn.Parameter(torch.log(torch.tensor([0.87], dtype=torch.float32)))
        self.log_vF = nn.Parameter(torch.log(torch.tensor([0.1], dtype=torch.float32)))

        # ------------------------------------------------------------------
        # Neural network (fully connected, tanh activations)
        # ------------------------------------------------------------------
        layers_in = []
        for in_f, out_f in zip(layers[:-1], layers[1:]):
            layers_in.append(nn.Linear(in_f, out_f))
            if out_f != layers[-1]:  # last layer linear
                layers_in.append(nn.Tanh())
        self.fnn = nn.Sequential(*layers_in)

        # ------------------------------------------------------------------
        # IRK (Butcher tableau): pre‑computed constants as buffers
        # ------------------------------------------------------------------
        tmp = np.loadtxt(f"Butcher_tableau/Butcher_IRK{self.steps_irk}.txt", ndmin=2, dtype=np.float32)
        weights = tmp[: self.steps_irk * self.steps_irk + self.steps_irk].reshape(self.steps_irk + 1, self.steps_irk)
        irk_alpha = weights[:-1]
        irk_beta = weights[-1:]
        self.register_buffer("irk_alpha", torch.tensor(irk_alpha))
        self.register_buffer("irk_beta", torch.tensor(irk_beta))
        self.register_buffer("irk_times", torch.tensor(tmp[self.steps_irk * self.steps_irk + self.steps_irk :]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scale(self, X: torch.Tensor) -> torch.Tensor:
        """Feature scaling to [-1, 1] (same as original)."""
        return 2.0 * (X - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0

    def _connect_to_PI_head(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through FNN & split into current/voltage."""
        out = self.fnn(self._scale(x))
        current_intermediate_predictions = out[:, : self.steps_irk]
        voltage_intermediate_predictions = out[:, self.steps_irk : 2 * self.steps_irk]
        S = x[:, 2:3]
        S_off = x[:, 3:4]
        dt = x[:, 4:5]
        return current_intermediate_predictions, voltage_intermediate_predictions, S, S_off, dt

    # ------------------------------------------------------------------
    # Physics residuals (eqs. (4) & (7) in the paper)
    # ------------------------------------------------------------------

    def _phys_params(self):
        """Returns exponentiated physical parameters with proper scaling."""
        L = torch.exp(self.log_L) * 1e-4
        RL = torch.exp(self.log_RL) * 1e-1
        C = torch.exp(self.log_C) * 1e-4
        RC = torch.exp(self.log_RC) * 1e-1
        Rdson = torch.exp(self.log_Rdson) * 1e-1
        Rload1 = torch.exp(self.log_Rload1)
        Rload2 = torch.exp(self.log_Rload2)
        Rload3 = torch.exp(self.log_Rload3)
        vin = torch.exp(self.log_vin) * 1e1
        vF = torch.exp(self.log_vF)
        return L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs both backward (U0) and forward (U1) passes in one go."""

        S = X[:, 2:3]
        dt = X[:, 3:4]

        out = self.fnn(self._scale(X))
        i_n, v_n, i_np1, v_np1 = self._PI_head(out, S, dt)
        return i_n, v_n, i_np1, v_np1

    def _PI_head(
        self,
        out_ffn: torch.Tensor,
        S: torch.Tensor,
        dt: torch.Tensor
        ):
        i_iterm_pred, v_iterm_pred = (
            out_ffn[:, : self.steps_irk],
            out_ffn[:, self.steps_irk : 2 * self.steps_irk],
        )
        
        L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF = self._phys_params()
        rload = torch.cat(
            [
                torch.ones(self.split_idx1, 1) * Rload1,
                torch.ones(self.split_idx2, 1) * Rload2,
                torch.ones(self.split_idx3, 1) * Rload3,
            ], dim=0
        )
        
        i_deriv = self.current_derivative_function(
            i_iterm_pred, v_iterm_pred, S, dt, L, RL, Rdson, vin, vF
        )
        v_deriv = self.voltage_derivative_function(
            i_iterm_pred, v_iterm_pred, S, dt, C, RC, rload, i_deriv
        )
            
        # Backward prediction
        i_n = i_iterm_pred - dt * (i_deriv @ self.irk_alpha.T)
        v_n = v_iterm_pred - dt * (v_deriv @ self.irk_alpha.T)
        # Forward prediction
        i_np1 = i_iterm_pred + dt * (i_deriv @ (self.irk_beta - self.irk_alpha).T)
        v_np1 = v_iterm_pred + dt * (v_deriv @ (self.irk_beta - self.irk_alpha).T)
        
        return (
            i_n, v_n, i_np1, v_np1
            )
        
    @staticmethod
    def current_derivative_function(
        i_iterm_pred: torch.Tensor,
        v_iterm_pred: torch.Tensor,
        S: torch.Tensor,
        dt: torch.Tensor,
        L: torch.Tensor,
        RL: torch.Tensor,
        Rdson: torch.Tensor,
        vin: torch.Tensor,
        vF: torch.Tensor
    ) -> torch.Tensor:
        """Computes the current derivative as per the physics model."""
        return - (
            (S * Rdson + RL) * i_iterm_pred 
            + v_iterm_pred 
            - S * vin + (1-S) * vF
        ) / L

        
    @staticmethod
    def voltage_derivative_function(
        i_iterm_pred: torch.Tensor,
        v_iterm_pred: torch.Tensor,
        S: torch.Tensor,
        dt: torch.Tensor,
        C: torch.Tensor,
        RC: torch.Tensor,
        rload: torch.Tensor,
        current_derivative_func_evaluated: torch.Tensor
    ) -> torch.Tensor:
        """Computes the voltage derivative as per the physics model."""
        return (
            C * RC * rload * current_derivative_func_evaluated + 
            rload * i_iterm_pred - 
            v_iterm_pred
        ) / (C * (RC + rload))

# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------


def compute_loss(
    preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y_n: torch.Tensor,
    y_np1: torch.Tensor,
) -> torch.Tensor:
    i_n_pred, v_n_pred, i_np1_pred, v_np1_pred = preds
    i_n, v_n = y_n[:, :1], y_n[:, 1:]
    i_np1, v_np1 = y_np1[:, :1], y_np1[:, 1:]
    
    loss = (
        torch.sum((i_n - i_n_pred) ** 2)
        + torch.sum((v_n - v_n_pred) ** 2)
        + torch.sum((i_np1 - i_np1_pred) ** 2)
        + torch.sum((v_np1 - v_np1_pred) ** 2)
    )
    return loss


def train_model(
    model: PINNBuck,
    x: torch.Tensor,
    xp1: torch.Tensor,
    params: OtherParams,
    device: torch.device,
) -> None:
    model.to(device)
    x, xp1, = x.to(device),xp1.to(device)

    # ------------------------------- Adam phase ----------------------------
    adam = torch.optim.Adam(model.parameters(), lr=params.adam_lr)
    # start = time.time()
    for it in range(params.adam_epochs):
        adam.zero_grad()
        x0 = x[:, :2]
        loss = compute_loss(
            model(x), 
            x0, 
            xp1
        )
        loss.backward()
        adam.step()

        if it % 500 == 0:
            # print(f"[Adam] iter={it:7d}, loss={loss.item():.3e}")
            L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF = model._phys_params()
            print(
                f"[Adam] iter={it:7d}, loss={loss.item():.3e}, "
                f"L={L.item():.3e}, RL={RL.item():.3e}, C={C.item():.3e}, "
                f"RC={RC.item():.3e}, Rdson={Rdson.item():.3e}, "
                f"Rload1={Rload1.item():.3e}, Rload2={Rload2.item():.3e}, "
                f"Rload3={Rload3.item():.3e}, Vin={vin.item():.3e}, VF={vF.item():.3e}"
            )

    # ------------------------------- LBFGS phase ---------------------------
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=params.lbfgs_epochs, tolerance_grad=1e-9)

    def closure():
        lbfgs.zero_grad()
        l = compute_loss(model(x), x0, xp1)
        l.backward()
        return l

    print("Starting L‑BFGS optimisation … (this may take a while)")
    lbfgs.step(closure)
    print("L‑BFGS finished.")


# -----------------------------------------------------------------------------
# Main experiment loop (mirrors original __main__)
# -----------------------------------------------------------------------------


def run(dataset_root: Path, device: torch.device):
    set_seed()
    params = OtherParams()

    q = 20
    layers = [4, 50, 50, 50, 50, 50, q * 2]

    results_f = open("Result.txt", "a", encoding="utf-8")

    test_idx = 0
    mat_path = dataset_root / f"buckSimulation_{test_idx}.mat"
    data = sio.loadmat(mat_path)

    # ----- load numpy arrays -----
    x_cur, x_volt, d_switch = data["CurrentInput"], data["VoltageInput"], data["Dswitch"]
    y_cur, y_volt = data["Current"], data["Voltage"]
    indicator, dt = data["forwaredBackwaredIndicator"], data["dt"]

    idx_forward = np.where(indicator[:, 0] == -2)[0]
    idx_backward = np.where(indicator[:, 0] == 2)[0]

    X = np.hstack([x_cur, x_volt, d_switch, dt])
    lb, ub = X.min(0), X.max(0)

    x = np.hstack(
        [
            x_cur[idx_forward],
            x_volt[idx_forward],
            d_switch[idx_forward],
            dt[idx_forward],
        ]
    ).reshape(-1, 4)
    u1 = np.hstack([y_cur[idx_backward], y_volt[idx_backward]]).reshape(-1, 2)

    # convert to torch tensors
    x_t = torch.tensor(x, dtype=torch.float32)
    u1_t = torch.tensor(u1, dtype=torch.float32)

    datapath = Path(r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases") / "buck_converter_Shuai_processed.h5"

    print(f"Loading data from {datapath} ...")
    subtransients = {}
    with h5py.File(datapath, "r") as f:
        # group is "ideal"
        gr = f["ideal"]
        for key in gr.keys():
            subtransients[key] = {}
            subtransients[key]["i"] = gr[key]["i"][:]
            subtransients[key]["v"] = gr[key]["v"][:]
            subtransients[key]["Dswitch"] = gr[key]["Dswitch"][:]
            subtransients[key]["dt"] = gr[key]["dt"][:]

    def data_loader(
        i: np.ndarray, v: np.ndarray, d_switch: np.ndarray, dt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns forward and backward vectors for the PINN."""
        # divide in input and output pairs
        x_cur = i[:-1].reshape(-1, 1)  # current input
        x_volt = v[:-1].reshape(-1, 1)  # voltage input
        y_cur = i[1:].reshape(-1, 1)  # current output
        y_volt = v[1:].reshape(-1, 1)  # voltage output

        d_switch = d_switch.reshape(-1, 1)  # duty cycle input
        dt = dt.reshape(-1, 1)  # time step input

        ## FORWARD: predict the next state
        X = np.hstack(
            [
                x_cur,
                x_volt,
                d_switch[:-1],
                dt[:-1],
            ]
        ).reshape(-1, 4)

        y = np.hstack([y_cur, y_volt]).reshape(-1, 2)
        return X, y

    # get forward and backward vectors for each subtransient
    subtransients_fwbw = {
        key: data_loader(
            subtransients[key]["i"],
            subtransients[key]["v"],
            subtransients[key]["Dswitch"],
            subtransients[key]["dt"],
        )
        for key in subtransients.keys()
    }

    for key in subtransients_fwbw.keys():
        print(f"Subtransient {key}: {subtransients_fwbw[key][0].shape[0]} samples")

    transient1_len = len(subtransients_fwbw["subtransient_1"][0])
    transient2_len = len(subtransients_fwbw["subtransient_2"][0])
    transient3_len = len(subtransients_fwbw["subtransient_3"][0])

    # concatenate all forward and backward vectors
    X = np.zeros((transient1_len + transient2_len + transient3_len, 4))
    y = np.zeros((transient1_len + transient2_len + transient3_len, 2))

    start_idx = 0
    for i, key in enumerate(subtransients_fwbw.keys()):
        X_i, y_i = subtransients_fwbw[key]
        end_idx = start_idx + X_i.shape[0]
        X[start_idx:end_idx, :] = X_i
        y[start_idx:end_idx, :] = y_i
        start_idx = end_idx

    lb, ub = X.min(0), X.max(0)

    # convert to torch tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    split_idx1 = transient1_len
    split_idx2 = transient2_len
    split_idx3 = transient3_len
    
    # split_idx1 = 240
    # split_idx2 = 240
    # split_idx3 = 240

    model = PINNBuck(
        layers, q, lb, ub, 
        split_idx1=split_idx1,
        split_idx2=split_idx2,  
        split_idx3=split_idx3
    )
    print(f"\n=== Training case {test_idx} ===")
    train_model(model, X_t, y_t, params, device)

    # ------------------- Evaluation & logging -----------------------
    with torch.no_grad():
        L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF = [
            p.exp().item()
            for p in [
                model.log_L,
                model.log_RL,
                model.log_C,
                model.log_RC,
                model.log_Rdson,
                model.log_Rload1,
                model.log_Rload2,
                model.log_Rload3,
                model.log_vin,
                model.log_vF,
            ]
        ]

    def rel_err(est, nom):
        return abs(est / nom * 100 - 100)

    vals = {
        "L": rel_err(L, params.nL),
        "RL": rel_err(RL, params.nRL),
        "C": rel_err(C, params.nC),
        "RC": rel_err(RC, params.nRC),
        "Rdson": rel_err(Rdson, params.nRdson),
        "Rload1": rel_err(Rload1, params.nRload1),
        "Rload2": rel_err(Rload2, params.nRload2),
        "Rload3": rel_err(Rload3, params.nRload3),
        "Vin": rel_err(vin, params.nVin),
        "VF": rel_err(vF, params.nVF),
    }
    mean_error = np.mean(list(vals.values()))

    results_f.write(
        f"buckSimulation_{test_idx}: mean:{mean_error:.2f}, "
        + ", ".join([f"{k}: {v:.2f}" for k, v in vals.items()])
        + "\n"
    )
    results_f.flush()
    print(f"Case {test_idx} finished. Mean relative error: {mean_error:.2f}%")

    results_f.close()


# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch PINN buck‑converter parameter estimator")
    parser.add_argument(
        "--data_root", type=str, default="Simulation_data", help="Path to folder with *.mat files"
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda")
    args = parser.parse_args()

    run(Path(args.data_root), torch.device(args.device))

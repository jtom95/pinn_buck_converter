"""
==================================================================================================
PyTorch re‑implementation of:
    S. Zhao, Y. Peng, Y. Zhang, and H. Wang,
    "Parameter Estimation of Power Electronic Converters with Physics‑informed Machine Learning",
    IEEE Trans. Power Electronics, 2022.

Original TF1.15 code by Shuai Zhao (Aalborg University).
Converted to PyTorch by ChatGPT, June 2025.

Key differences from the original implementation
-------------------------------------------------
* TensorFlow placeholders/sessions have been replaced with idiomatic
  PyTorch `Dataset`, `DataLoader`, and `nn.Module` constructs.
* Optimisation now relies on `torch.optim.Adam` followed by
  `torch.optim.LBFGS` (PyTorch’s built‑in variant) instead of
  `tf.contrib.opt.ScipyOptimizerInterface`.
* All trainable physical parameters are stored as \log‑parameters
  (`nn.Parameter`) to enforce positivity via `torch.exp`.
* The implicit Runge–Kutta (IRK) Butcher tableau is loaded with NumPy
  and cached as `torch.Tensor` buffers (`register_buffer`).
* Code has been modularised so that the physics‑constrained residuals
  (`net_u0`, `net_u1`) are computed inside the module’s `forward` call.
* A simple CLI (`python pinn_buck_converter_pytorch.py --device cuda`) is
  provided for convenience.

Python ≥3.8 and PyTorch ≥2.0 are assumed.
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
        q: int,
        lb: np.ndarray,
        ub: np.ndarray,
        split_idx1: int,
        split_idx2: int,
        split_idx3: int,
    ) -> None:
        super().__init__()

        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)
        self.q = max(q, 1)
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
        tmp = np.loadtxt(f"Butcher_tableau/Butcher_IRK{q}.txt", ndmin=2, dtype=np.float32)
        weights = tmp[: q * q + q].reshape(q + 1, q)
        irk_alpha = weights[:-1]
        irk_beta = weights[-1:]
        self.register_buffer("irk_alpha", torch.tensor(irk_alpha))
        self.register_buffer("irk_beta", torch.tensor(irk_beta))
        self.register_buffer("irk_times", torch.tensor(tmp[q * q + q :]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scale(self, X: torch.Tensor) -> torch.Tensor:
        """Feature scaling to [-1, 1] (same as original)."""
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def _net_uv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through FNN & split into current/voltage."""
        out = self.fnn(self._scale(x))
        u = out[:, : self.q]
        v = out[:, self.q : 2 * self.q]
        return u, v

    # ------------------------------------------------------------------
    # Physics residuals (eqs. (4) & (7) in the paper)
    # ------------------------------------------------------------------

    def _net_u0(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF = self._phys_params()

        y = x[:, 2:3]
        y_off = x[:, 3:4]
        dt = x[:, 4:5]

        rload = torch.cat(
            [
                torch.ones(self.split_idx1 * 2, 1) * Rload1,
                torch.ones(self.split_idx2 * 2, 1) * Rload2,
                torch.ones(self.split_idx3 * 2, 1) * Rload3,
            ],
            dim=0,
        )

        u, v = self._net_uv(x)

        f_u = -((y * (RL + Rdson)) * u + (y_off * RL) * u + v - y * vin + y_off * vF) / L
        u0 = u - dt * (f_u @ self.irk_alpha.T)
        

        f_v = (C * RC * rload * f_u + rload * u - v) / (C * (RC + rload))
        v0 = v - dt * (f_v @ self.irk_alpha.T)
        return u0, v0

    def _net_u1(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF = self._phys_params()

        y = x[:, 2:3]
        y_off = x[:, 3:4]
        dt = x[:, 4:5]

        rload = torch.cat(
            [
                torch.ones(self.split_idx1 * 2, 1) * Rload1,
                torch.ones(self.split_idx2 * 2, 1) * Rload2,
                torch.ones(self.split_idx3 * 2, 1) * Rload3,
            ],
            dim=0,
        )

        u, v = self._net_uv(x)
        
        

        f_u = -((y * (RL + Rdson)) * u + (y_off * RL) * u + v - y * vin + y_off * vF) / L
        u1 = u + dt * (f_u @ (self.irk_beta - self.irk_alpha).T)

        f_v = (C * RC * rload * f_u + rload * u - v) / (C * (RC + rload))
        v1 = v + dt * (f_v @ (self.irk_beta - self.irk_alpha).T)
        return u1, v1

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
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs both backward (U0) and forward (U1) passes in one go."""
        u0_pred, v0_pred = self._net_u0(x0)
        u1_pred, v1_pred = self._net_u1(x1)
        return u0_pred, v0_pred, u1_pred, v1_pred


# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------


def compute_loss(
    preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    u0: torch.Tensor,
    u1: torch.Tensor,
) -> torch.Tensor:
    u0_pred, v0_pred, u1_pred, v1_pred = preds
    loss = (
        torch.sum((u0[:, :1] - u0_pred) ** 2)
        + torch.sum((u0[:, 1:] - v0_pred) ** 2)
        + torch.sum((u1[:, :1] - u1_pred) ** 2)
        + torch.sum((u1[:, 1:] - v1_pred) ** 2)
    )
    return loss


def train_model(
    model: PINNBuck,
    x0: torch.Tensor,
    u0: torch.Tensor,
    x1: torch.Tensor,
    u1: torch.Tensor,
    params: OtherParams,
    device: torch.device,
) -> None:
    model.to(device)
    x0, u0, x1, u1 = x0.to(device), u0.to(device), x1.to(device), u1.to(device)

    # ------------------------------- Adam phase ----------------------------
    adam = torch.optim.Adam(model.parameters(), lr=params.adam_lr)
    for it in range(params.adam_epochs):
        adam.zero_grad()
        loss = compute_loss(model(x0, x1), u0, u1)
        loss.backward()
        adam.step()

        if it % 500 == 0:
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
        l = compute_loss(model(x0, x1), u0, u1)
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
    layers = [5, 50, 50, 50, 50, 50, q * 2]

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

    X = np.hstack([x_cur, x_volt, d_switch, 1 - d_switch, dt])
    lb, ub = X.min(0), X.max(0)

    x0 = np.hstack(
        [
            x_cur[idx_forward],
            x_volt[idx_forward],
            d_switch[idx_forward],
            1 - d_switch[idx_forward],
            dt[idx_forward],
        ]
    ).reshape(-1, 5)
    u0 = np.hstack([y_cur[idx_forward], y_volt[idx_forward]]).reshape(-1, 2)
    x1 = np.hstack(
        [
            x_cur[idx_backward],
            x_volt[idx_backward],
            d_switch[idx_backward],
            1 - d_switch[idx_backward],
            dt[idx_backward],
        ]
    ).reshape(-1, 5)
    u1 = np.hstack([y_cur[idx_backward], y_volt[idx_backward]]).reshape(-1, 2)

    # convert to torch tensors
    x0_t = torch.tensor(x0, dtype=torch.float32)
    u0_t = torch.tensor(u0, dtype=torch.float32)
    x1_t = torch.tensor(x1, dtype=torch.float32)
    u1_t = torch.tensor(u1, dtype=torch.float32)

    model = PINNBuck(layers, q, lb, ub, split_idx1=120, split_idx2=120, split_idx3=120)
    print(f"\n=== Training case {test_idx} ===")
    train_model(model, x0_t, u0_t, x1_t, u1_t, params, device)

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

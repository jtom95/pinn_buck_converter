import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

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
        tmp = np.loadtxt(
            f"Butcher_tableau/Butcher_IRK{self.steps_irk}.txt", ndmin=2, dtype=np.float32
        )
        weights = tmp[: self.steps_irk * self.steps_irk + self.steps_irk].reshape(
            self.steps_irk + 1, self.steps_irk
        )
        irk_alpha = weights[:-1]
        irk_beta = weights[-1:]
        self.register_buffer("irk_alpha", torch.tensor(irk_alpha))
        self.register_buffer("irk_beta", torch.tensor(irk_beta))
        self.register_buffer(
            "irk_times", torch.tensor(tmp[self.steps_irk * self.steps_irk + self.steps_irk :])
        )

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

    def _PI_head(self, out_ffn: torch.Tensor, S: torch.Tensor, dt: torch.Tensor):
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
            ],
            dim=0,
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

        return (i_n, v_n, i_np1, v_np1)

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
        vF: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the current derivative as per the physics model."""
        return -((S * Rdson + RL) * i_iterm_pred + v_iterm_pred - S * vin + (1 - S) * vF) / L

    @staticmethod
    def voltage_derivative_function(
        i_iterm_pred: torch.Tensor,
        v_iterm_pred: torch.Tensor,
        S: torch.Tensor,
        dt: torch.Tensor,
        C: torch.Tensor,
        RC: torch.Tensor,
        rload: torch.Tensor,
        current_derivative_func_evaluated: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the voltage derivative as per the physics model."""
        return (
            C * RC * rload * current_derivative_func_evaluated + rload * i_iterm_pred - v_iterm_pred
        ) / (C * (RC + rload))

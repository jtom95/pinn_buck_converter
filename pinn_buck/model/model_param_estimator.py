import numpy as np
import torch
from torch import nn

from pinn_buck.config import Parameters
from pinn_buck.parameter_transformation import make_log_param, reverse_log_param

from abc import ABC, abstractmethod
from typing import Tuple


from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from ..config import Parameters
from ..io import Measurement


import numpy as np
import torch
from torch import nn


def measurement_to_tensors(meas: Measurement, device=None) -> Tuple[torch.Tensor, torch.LongTensor]:
    """
    Convert a Measurement into
        X        : (N, 4) float32 tensor  [i, v, D, dt]
        run_idx  : (N,)  int64   tensor  each row ∈ {0,1,2}
    """
    X_parts, run_parts = [], []
    for k, tr in enumerate(meas.transients):
        i, v, D, dt = tr.i, tr.v, tr.D, tr.dt
        X_parts.append(np.hstack([i[:-1, None], v[:-1, None], D[:-1, None], dt[:-1, None]]))
        run_parts.append(np.full(len(i) - 1, k, dtype=np.int64))

    X_np = np.vstack(X_parts).astype(np.float32)
    run_idx_np = np.concatenate(run_parts)

    device = torch.device(device or "cpu") 
    return (
        torch.as_tensor(X_np, device=device, dtype=torch.float32),
        torch.as_tensor(run_idx_np, device=device, dtype=torch.long),  # explicit dtype
    )


class BuckParamEstimator__(nn.Module):
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

    # def _scale(self, x: torch.Tensor):
    #     return 2 * (x - self.lb) / (self.ub - self.lb) - 1

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


class BuckParamEstimator(BuckParamEstimator__):
    """Physics‑informed NN for parameter estimation in a buck converter."""

    def _make_rload_vector(self, N: int, device: torch.device, p: Parameters):
        """
        Return an (N,1) tensor with [Rload1]*k1 + [Rload2]*k2 + [Rload3]*k3
        where:
            k1 = min(N,  s1)
            k2 = min(max(N - s1, 0), s2)
            k3 = max(N - s1 - s2, 0)   (clipped to ≤ s3 but N never exceeds s1+s2+s3)
        """
        # how many rows belong to each load in THIS batch
        k1 = min(N, self.s1)
        k2 = min(max(N - self.s1, 0), self.s2)
        k3 = max(N - self.s1 - self.s2, 0)

        parts = []
        if k1:
            parts.append(torch.ones((k1, 1), device=device) * p.Rload1)
        if k2:
            parts.append(torch.ones((k2, 1), device=device) * p.Rload2)
        if k3:
            parts.append(torch.ones((k3, 1), device=device) * p.Rload3)

        return torch.cat(parts, dim=0)  # (N,1)

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        i_n, v_n = X[:, 0:1], X[:, 1:2]
        S, dt = X[:, 2:3], X[:, 3:4]

        i_np1, v_np1 = y[:, 0:1], y[:, 1:2]

        p = self._physical()
        rload = self._make_rload_vector(N=X.size(0), device=X.device, p=p)

        # forward prediction using RK4
        i_np1_pred, v_np1_pred = self._rk4_step(i_n, v_n, S, dt, p, rload, sign=+1)

        # backward prediction using RK4
        i_n_pred, v_n_pred = self._rk4_step(i_np1, v_np1, S, dt, p, rload, sign=-1)
        return i_n_pred, v_n_pred, i_np1_pred, v_np1_pred

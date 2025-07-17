import numpy as np
import torch
from torch import nn

from pinn_buck.config import Parameters
from pinn_buck.parameter_transformation import make_log_param, reverse_log_param

from abc import ABC, abstractmethod
from typing import Tuple


from __future__ import annotations  # Python ≥3.10: allows | in type hints
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from ..config import Parameters
from ..io import Measurement


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


class BaseBuckEstimator(nn.Module, ABC):
    """
    Base class for any buck-converter estimator.

    forward(X, run_idx=None, **kw)
        X        : (N, 4)  rows [i, v, D, Δt]
        run_idx  : (N,) int64 in {0,1,2} telling which transient each row belongs to.
                   If None we assume rows are ordered 0→1→2 (sequential batch).
    """

    # ------------------------------------------------------------------ init
    def __init__(self, param_init: Parameters) -> None:
        super().__init__()

        # ---- trainable log-parameters ------------------------------------
        log_p = make_log_param(param_init)
        for name, val in log_p._asdict().items():  # e.g. "L", "RL", …
            self.register_parameter(f"log_{name}", nn.Parameter(val.clone(), True))

    # ------------------------------------------------ parameter utilities ---
    @property
    def _log_keys(self):
        return [n for n, _ in self.named_parameters() if n.startswith("log_")]

    @property
    def logparams(self) -> Parameters:
        """Return **current** log-space parameters as a named-tuple."""
        return Parameters(**{k[4:]: getattr(self, k) for k in self._log_keys})

    def _physical(self) -> Parameters:
        """Convert log-space -> physical-space (vectorised exp)."""
        return reverse_log_param(self.logparams)

    def get_estimates(self) -> Parameters:
        """Detach and cast to Python floats — handy for printing/logging."""
        return Parameters(**{k: v.item() for k, v in self._physical()._asdict().items()})

    # ---------------------------------------- R_load helpers --------------
    def _rload_from_seq(self, N: int, device: torch.device, p: Parameters, run_lengths):
        """Rows are in transient order 0→1→2."""
        l1, l2, l3 = run_lengths
        k1 = min(N, l1)
        k2 = min(max(N - l1, 0), l2)
        k3 = max(N - l1 - l2, 0)

        parts = []
        if k1:
            parts.append(torch.ones((k1, 1), device=device) * p.Rload1)  
        if k2:
            parts.append(torch.ones((k2, 1), device=device) * p.Rload2)  
        if k3:
            parts.append(torch.ones((k3, 1), device=device) * p.Rload3)  
        return torch.cat(parts, 0)

    @staticmethod
    def _rload_from_idx(run_idx: torch.LongTensor, p: Parameters):
        """Rows may be arbitrarily shuffled; use run_idx to look up Rload."""
        lookup = torch.tensor(
            [p.Rload1, p.Rload2, p.Rload3],
            dtype=torch.float32, device=run_idx.device,
        )
        return lookup[run_idx].unsqueeze(-1)

    # ------------------------------------------- single RK-4 step ----------
    @staticmethod
    def _rk4_step(i, v, D, dt, p: Parameters, rload, sign: int = +1):
        """
        One Runge–Kutta-4 step of the buck-converter ODE.

        * `sign = +1` → forward step  (n → n+1)
        * `sign = -1` → backward step (n+1 → n)
        """
        dh = dt * sign

        def rhs(i_, v_):
            di = -((D * p.Rdson + p.RL) * i_ + v_ - D * p.Vin + (1 - D) * p.VF) / p.L
            dv = (p.C * p.RC * rload * di + rload * i_ - v_) / (p.C * (p.RC + rload))
            return di, dv

        k1_i, k1_v = rhs(i, v)
        k2_i, k2_v = rhs(i + 0.5 * dh * k1_i, v + 0.5 * dh * k1_v)
        k3_i, k3_v = rhs(i + 0.5 * dh * k2_i, v + 0.5 * dh * k2_v)
        k4_i, k4_v = rhs(i + dh * k3_i, v + dh * k3_v)

        i_new = i + dh * (k1_i + 2 * k2_i + 2 * k3_i + k4_i) / 6.0
        v_new = v + dh * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
        return i_new, v_new

    # ------------------------------------------------------- abstract fwd --
    @abstractmethod
    def forward(self, X: torch.Tensor, run_idx: torch.LongTensor, **kwargs):
        """
        Parameters
        ----------
        X : (N, 4) tensor — rows xₙ = [iₙ, vₙ, Dₙ, Δtₙ].
            Sub-classes choose how to slice/reshape X before calling `_rk4_step`.
            
        run_idx : (N,) tensor of int32 in {0,1,2} indicating which transient each row belongs to.

        Returns
        -------
        Any : typically a tuple `(preds, targets)` compatible with the loss.
        """
        ...


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

    # ---------- main forward -----------------------------------------------

class BuckParamEstimatorTriplets(BuckParamEstimator__):
    def forward(self, X: torch.Tensor):
        """
        Build disjoint triplets (n-1,n,n+1) **within each transient**.

        Each run length (s1,s2,s3) may not be a multiple of 3; the final
        1-or-2 samples of that run are dropped (only inside that run).

        Returns
        -------
        preds   : (M,4)  [i_{n+1}^pred, v_{n+1}^pred, i_n^pred, v_n^pred]
        targets : (M,4)  [i_{n+1}^obs , v_{n+1}^obs , i_n^obs , v_n^obs ]
        M       : total #triplets  =  ⌊s1/3⌋+⌊s2/3⌋+⌊s3/3⌋
        """
        device = X.device
        p = self._physical()

        # ------------------------------------------------------------------
        # 1. split X into three runs, trim each run to a multiple of 3
        # ------------------------------------------------------------------
        runs_len = [self.s1, self.s2, self.s3]
        rload_val = [p.Rload1, p.Rload2, p.Rload3]

        X_list = []  # trimmed observation rows
        rload_ls = []  # matching Rload rows

        start = 0
        for L, Rval in zip(runs_len, rload_val):
            end = start + L
            X_run = X[start:end]  # (L,4)

            T = (L // 3) * 3  # largest multiple of 3 ≤ L
            if T:  # skip run if <3 samples
                X_run = X_run[:T]  # (T,4)
                X_list.append(X_run)

                r_vec = torch.ones((T, 1), device=device) * Rval
                rload_ls.append(r_vec)
            start = start + L

        # concat all trimmed runs
        if not X_list:  # every run <3 rows?
            raise RuntimeError("No complete triplet found in batch.")

        X_all = torch.cat(X_list, dim=0)  # (N_tot,4)
        rload_all = torch.cat(rload_ls, dim=0)  # (N_tot,1)
        T_trip = X_all.shape[0] // 3  # #triplets

        # ------------------------------------------------------------------
        # 2. reshape into triplets (T,3, ...)
        # ------------------------------------------------------------------
        X_trip = X_all.view(T_trip, 3, 4)
        rload_trip = rload_all.view(T_trip, 3, 1)

        X_prev, X_curr, X_next = X_trip.unbind(dim=1)
        r_prev, r_curr, _ = rload_trip.unbind(dim=1)

        # unpack
        i_prev, v_prev = X_prev[:, 0:1], X_prev[:, 1:2]
        S_prev, dt_prev = X_prev[:, 2:3], X_prev[:, 3:4]

        i_curr, v_curr = X_curr[:, 0:1], X_curr[:, 1:2]
        S_curr, dt_curr = X_curr[:, 2:3], X_curr[:, 3:4]

        i_next, v_next = X_next[:, 0:1], X_next[:, 1:2]

        # ------------------------------------------------------------------
        # 3. two batched RK4 steps
        # ------------------------------------------------------------------
        i_curr_pred, v_curr_pred = self._rk4_step(i_prev, v_prev, S_prev, dt_prev, p, r_prev, sign=+1)
        i_next_pred, v_next_pred = self._rk4_step(i_curr, v_curr, S_curr, dt_curr, p, r_curr, sign=+1)

        # ------------------------------------------------------------------
        # 4. stack to (M,4)
        # ------------------------------------------------------------------
        targets = torch.cat([i_next, v_next, i_curr, v_curr], dim=1)  # (M,4)
        preds = torch.cat([i_next_pred, v_next_pred, i_curr_pred, v_curr_pred], dim=1)  # (M,4)

        return preds, targets


import torch
from torch import Tensor

from pinn_buck.config import Parameters
from pinn_buck.parameter_transformation import make_log_param, reverse_log_param

import torch
from pinn_buck.config import Parameters
from pinn_buck.parameter_transformation import make_log_param, reverse_log_param


# ---------------------------------------------------------------------
# Cached incremental-residual PINN
# ---------------------------------------------------------------------
class BuckEstimatorCached(BuckParamEstimator__):
    """
    Minimises first-difference residuals     r_Δ = (ŷ_{n+1}-ŷ_n) - (y_{n+1}-y_n)

    • Caches the previous prediction → keeps   L-2   samples out of   L
      (vs. only ⌊L/3⌋ with independent triplets).
    • Keeps transients separate, so Δ-residuals never combine data with
      different R_load values.
    • forward() returns (Δŷ, Δy)    each of shape (M, 2)
      with M = Σ_run (L_run-2).
    """

    def forward(self, X: torch.Tensor):
        device = X.device
        p: Parameters = self._physical()  # current parameter estimates

        runs = torch.split(X, [self.s1, self.s2, self.s3])
        rvals = [p.Rload1, p.Rload2, p.Rload3]

        delta_pred_parts, delta_obs_parts = [], []

        for X_run, R in zip(runs, rvals):
            L = X_run.size(0)
            if L < 3:  # need ≥3 samples to form one Δ-residual
                continue

            # unpack
            i, v = X_run[:, 0:1], X_run[:, 1:2]
            S, dt = X_run[:, 2:3], X_run[:, 3:4]

            # vector of constant R_load for this run (length L-1 for the RK4 step)
            r_vec = torch.ones((L - 1, 1), device=device)*R

            # one-step prediction ŷ_{n+1|n},  n = 0 … L-2
            i_hat_np1, v_hat_np1 = self._rk4_step(
                i[:-1], v[:-1], S[:-1], dt[:-1], p, r_vec, sign=+1
            )  # tensors (L-1, 1)

            # form first differences on indices 1 … L-2  (drop first + last)
            delta_i_hat = i_hat_np1[1:] - i_hat_np1[:-1]  # (L-2, 1)
            delta_v_hat = v_hat_np1[1:] - v_hat_np1[:-1]

            delta_i = i[2:] - i[1:-1]  # (L-2, 1)
            delta_v = v[2:] - v[1:-1]

            delta_pred_parts.append(torch.cat([delta_i_hat, delta_v_hat], dim=1))  # (L-2, 2)
            delta_obs_parts.append(torch.cat([delta_i, delta_v], dim=1))

        if not delta_pred_parts:  # all runs too short?
            raise RuntimeError("No delta_-residuals could be formed (all runs < 3 rows).")

        delta_pred = torch.cat(delta_pred_parts, dim=0)  # (M, 2)
        delta_obs = torch.cat(delta_obs_parts, dim=0)
        return delta_pred, delta_obs


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

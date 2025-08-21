import numpy as np
import torch
from torch import nn

from pinn_buck.parameters.parameter_class import Parameters
from pinn_buck.parameter_transformation import make_log_param, reverse_log_param

from abc import ABC, abstractmethod
from typing import Tuple


from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from ..parameters.parameter_class import Parameters
from ..io import Measurement


import numpy as np
import torch
from torch import nn


from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any, Literal


class BaseBuckEstimator(nn.Module, ABC):
    """
    Base class for any buck-converter estimator.

    forward(X, **kw)
        X        : (B, T, 4) tensor of inputs
        kw       : additional keyword arguments for the forward pass
        returns  : (B, T, p) tensor of outputs, where p is the number of outputs (e.g., 2 for i and v or 4 if we have both forward and backward predictions)

    where B is the batch size and T is the number of different transients. This is a simple and intuitive way to organize and represent the different transients in the data. However the
    limitation is that the model can only handle transients of the same length.

    the dimension of 4 corresponds to (i, v, D, dt).
    """

    def __init__(
        self,
        param_init: Parameters,
    ) -> None:
        super().__init__()
        self._initialize_log_parameters(param_init)

    def _initialize_log_parameters(self, param_init: Parameters):
        log_params = make_log_param(param_init)

        self.log_L = nn.Parameter(log_params.L, requires_grad=True)
        self.log_RL = nn.Parameter(log_params.RL, requires_grad=True)
        self.log_C = nn.Parameter(log_params.C, requires_grad=True)
        self.log_RC = nn.Parameter(log_params.RC, requires_grad=True)
        self.log_Rdson = nn.Parameter(log_params.Rdson, requires_grad=True)
        self.log_Rloads = nn.ParameterList(
            [nn.Parameter(r, requires_grad=True) for r in log_params.Rloads]
        )
        self.log_Vin = nn.Parameter(log_params.Vin, requires_grad=True)
        self.log_VF = nn.Parameter(log_params.VF, requires_grad=True)

    # ----------------------------- helpers -----------------------------

    def _physical(self) -> Parameters:
        """Return current parameters in physical units (inverse scaling)."""
        return reverse_log_param(self.logparams)

    @property
    def logparams(self) -> Parameters:
        """Return current log‑space parameters."""
        return Parameters(
            L=self.log_L,
            RL=self.log_RL,
            C=self.log_C,
            RC=self.log_RC,
            Rdson=self.log_Rdson,
            Rloads=[rload for rload in self.log_Rloads],
            Vin=self.log_Vin,
            VF=self.log_VF,
        )

    def get_estimates(self) -> Parameters:
        """Return current parameters in physical units."""
        params = self._physical()
        return Parameters(
            L=params.L.item(),
            RL=params.RL.item(),
            C=params.C.item(),
            RC=params.RC.item(),
            Rdson=params.Rdson.item(),
            Rloads=[rload.item() for rload in params.Rloads],
            Vin=params.Vin.item(),
            VF=params.VF.item(),
        )

    def _logparam_name_map(self) -> list[tuple[str, str]]:
        """
        Returns ordered pairs  (display, stored)
        e.g. ("L", "log_L"), ("Rload1", "log_Rloads.0"), ...
        """
        mapping = []
        for disp, _ in self.logparams.iterator():
            if disp.startswith("Rload"):
                idx = int(disp[5:]) - 1
                stored = f"log_Rloads.{idx}"
            else:
                stored = f"log_{disp}"
            mapping.append((disp, stored))
        return mapping

    # ---------------------- physics right‑hand sides -------------------
    @staticmethod
    def _di(i_k, v_k, S, p: Parameters):
        return -((S * p.Rdson + p.RL) * i_k + v_k - S * p.Vin + (1 - S) * p.VF) / p.L

    @staticmethod
    def _dv(i_k, v_k, S, p: Parameters, rload, di):
        return (p.C * p.RC * rload * di + rload * i_k - v_k) / (p.C * (p.RC + rload))

    # ------------------------------- forward --------------------------
    @staticmethod
    def _rk4_step(i, v, D, dt, p: Parameters, sign=+1):
        """
        One Runge–Kutta-4 step of the buck-converter ODE.

        * `sign = +1` → forward step  (n → n+1)
        * `sign = -1` → backward step (n+1 → n)

        Vectorized for tensors shape [..., 1].
        i, v, D, dt have shape [B, T, 1], where B is the batch size and T is the number of transients.
        p.Rloads is a list [Rload_0, Rload_1, ..., Rload_T-1]

        Parameters:
        - i: current at time n, shape [B, T, 1]
        - v: voltage at time n, shape [B, T, 1]
        - D: duty cycle at time n, shape [B, T, 1]
        - dt: time step at time n, shape [B, T, 1]
        - p: Parameters object containing the physical parameters of the buck converter
        - sign: +1 for forward step, -1 for backward step
        Returns:
        - i_new: current at time n±1, shape [B, T, 1]
        - v_new: voltage at time n±1, shape [B, T, 1]
        """
        dh = dt * sign

        # Build rload tensor of shape [1, 1] for broadcasting
        rload = torch.stack(p.Rloads).view(1, -1)

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

    @abstractmethod
    def forward(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (N, T, 4) tensor — rows xₙ = [iₙ, vₙ, Dₙ, Δtₙ].
            Sub-classes choose how to slice/reshape X before calling `_rk4_step`.

        Returns
        -------
        Pred : torch.Tensor predictions.
        """
        ...
        
    @abstractmethod
    def targets(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the target values for the given input tensor X. The targets must
        be generated to match the logic and the shape of the forward pass.
        Parameters
        ----------
        X : (N, T, 4) tensor — rows xₙ = [iₙ, vₙ, Dₙ, Δtₙ].
            Sub-classes choose how to slice/reshape X before calling `_rk4_step`.
        Returns
        -------
        targets : torch.Tensor
            The target values for the given input tensor X.
        """
        ...


class BuckParamEstimator(BaseBuckEstimator):
    """Physics‑informed NN for parameter estimation in a buck converter."""

    def forward(self, X: torch.Tensor):
        """
        X: [batch, n_transients, 4] -> (i_n, v_n, D, dt)

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
        pred: Forward prediction of the buck converter state at time n+1. Shape (B, T, 2)
        """

        i, v = X[..., 0], X[..., 1]
        D, dt = X[..., 2], X[..., 3]

        p = self._physical()

        # Forward and backward predictions (vectorized)
        pred = self._rk4_step(i[:-1], v[:-1], D[:-1], dt[:-1], p, sign=+1)
        # note that in the bck_pred we use D[:-1] and dt[:-1], this is because we want to predict the
        # voltages and currents at the time n, given their values after the dt time at time n+1.
        # transform the tuple into a tensor by stacking on the last dimension
        pred = torch.stack(pred, dim=-1)  # shape (B, T, 2)
        return pred

    def targets(self, X:torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the target values for the given input tensor X. The targets must
        be generated to match the logic and the shape of the forward pass.

        Parameters
        ----------
        X : (N, T, 4) tensor — rows xₙ = [iₙ, vₙ, Dₙ, Δtₙ].
            Sub-classes choose how to slice/reshape X before calling `_rk4_step`.

        Returns
        -------
        targets : torch.Tensor
            The target values for the given input tensor X.
        """
        return X[1:, :, :2].clone().detach()
    

class BuckParamEstimatorFwdBck(BaseBuckEstimator):
    """Physics‑informed NN for parameter estimation in a buck converter."""

    def forward(self, X: torch.Tensor):
        """
        X: [batch, n_transients, 4] -> (i_n, v_n, D, dt)

        Returns:
        -------
        torch.tensor (2, B, T, 2)

        along the first axis of the tensor:
        fwd_pred: Forward prediction of the buck converter state at time n+1. Shape (B, T, 2)
        bck_pred: Backward prediction of the buck converter state at time n-1. Shape (B, T, 2)
        """

        i, v = X[..., 0], X[..., 1]
        D, dt = X[..., 2], X[..., 3]

        p = self._physical()

        # Forward and backward predictions (vectorized)
        fwd_pred = self._rk4_step(i[:-1], v[:-1], D[:-1], dt[:-1], p, sign=+1)
        bck_pred = self._rk4_step(i[1:], v[1:], D[:-1], dt[:-1], p, sign=-1)
        # note that in the bck_pred we use D[:-1] and dt[:-1], this is because we want to predict the
        # voltages and currents at the time n, given their values after the dt time at time n+1. However,
        # D and dt refer to the time step from n to n+1, so we need to use the same D and dt as in the forward prediction.

        # transform the tuple into a tensor by stacking on the last dimension
        fwd_pred = torch.stack(fwd_pred, dim=-1)  # shape (B, T, 2)
        bck_pred = torch.stack(bck_pred, dim=-1)  # shape (B, T, 2)

        return torch.stack((fwd_pred, bck_pred), dim=0)  # shape (2, B, T, 2)

    def targets(self, X, **kwargs) -> torch.Tensor:
        """
        Returns the target values for the given input tensor X. The targets must
        be generated to match the logic and the shape of the forward pass.

        Parameters
        ----------
        X : (N, T, 4) tensor — rows xₙ = [iₙ, vₙ, Dₙ, Δtₙ].
            Sub-classes choose how to slice/reshape X before calling `_rk4_step`.

        Returns
        -------
        targets : torch.Tensor
            The target values for the given input tensor X, must be both forward and backward predictions.
        """
        target_fwd = X[1:, :, :2].clone().detach()
        target_bck = X[:-1, :, :2].clone().detach()
        return torch.stack((target_fwd, target_bck), dim=0)  # shape (2, B, T, 2)

import numpy as np
import torch
from torch import nn
import re

from pinn_buck.parameters.parameter_class import Parameters
from pinn_buck.parameter_transformation import make_log_param, reverse_log_param

from abc import ABC, abstractmethod
from typing import Tuple


from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from ..parameters.parameter_class import Parameters
from .system_state_class import States
from ..io import Measurement
from .system_dynamics import SystemDynamics
from ..buck_converter_classes import BuckConverterDynamics

import numpy as np
import torch
from torch import nn


from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any, Literal, Mapping, Callable, Dict


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
        self.system_dynamics: SystemDynamics = self._define_system_dynamics()

    def _initialize_log_parameters(self, param_init: Parameters):
        log_params = make_log_param(param_init)

        self.log_L    = nn.Parameter(torch.as_tensor(log_params.L,    dtype=torch.float32))
        self.log_RL   = nn.Parameter(torch.as_tensor(log_params.RL,   dtype=torch.float32))
        self.log_C    = nn.Parameter(torch.as_tensor(log_params.C,    dtype=torch.float32))
        self.log_RC   = nn.Parameter(torch.as_tensor(log_params.RC,   dtype=torch.float32))
        self.log_Rdson= nn.Parameter(torch.as_tensor(log_params.Rdson,dtype=torch.float32))
        self.log_Vin  = nn.Parameter(torch.as_tensor(log_params.Vin,  dtype=torch.float32))
        self.log_VF   = nn.Parameter(torch.as_tensor(log_params.VF,   dtype=torch.float32))

        # lists
        self.log_Rloads = nn.ParameterList(
            [nn.Parameter(torch.as_tensor(r, dtype=torch.float32)) for r in log_params.Rloads]
        )

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
            L=params.L,
            RL=params.RL,
            C=params.C,
            RC=params.RC,
            Rdson=params.Rdson,
            Rloads=[rload for rload in params.Rloads],
            Vin=params.Vin,
            VF=params.VF,
        )

    def _logparam_name_map(self) -> list[tuple[str, str]]:
        """
        Returns ordered pairs  (display, stored)
        e.g. ("L", "log_L"), ("Rload1", "log_Rloads.0"), ...
        """
        mapping = []
        for disp, _ in self.logparams.iterator():
            if disp.startswith("Rload"):
                # find numeric character at the END of the string
                match = re.search(r"\d+$", disp)
                if match:
                    idx = int(match.group(0)) - 1
                stored = f"log_Rloads.{idx}"
            else:
                stored = f"log_{disp}"
            mapping.append((disp, stored))
        return mapping

    def _rk4_step(
        self, 
        time_steps: torch.Tensor,
        states: States,
        params: Parameters,
        controls: Optional[Dict[str, torch.Tensor] | None] = None,
        sign: int = +1,
    ) -> States:
        """
        One Runge–Kutta-4 step of the buck-converter ODE.

        * `sign = +1` → forward step  (n → n+1)
        * `sign = -1` → backward step (n+1 → n)
        """

        dh = time_steps * float(sign) # step size, signed

        k1 = self.system_dynamics.dynamics(states, params, controls)
        k2 = self.system_dynamics.dynamics(states + (0.5 * dh) * k1, params, controls)
        k3 = self.system_dynamics.dynamics(states + (0.5 * dh) * k2, params, controls)
        k4 = self.system_dynamics.dynamics(states + dh * k3, params, controls)

        return states + (dh / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    @abstractmethod
    def _define_system_dynamics(self) -> SystemDynamics:
        """Define the system dynamics class."""
        ...

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
    def _define_system_dynamics(self) -> SystemDynamics:
        return BuckConverterDynamics()

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

        input_states = States({"i": i[:-1], "v": v[:-1]})
        controls = {"D": D[:-1]}
        time_steps = dt[:-1]
        
        params = self._physical().expand_torch_sequences()

        # Forward and backward predictions (vectorized)
        out_states = self._rk4_step(
            time_steps=time_steps, 
            states=input_states, 
            params=params, 
            controls=controls, 
            sign=+1
            )

        # transform the tuple into a tensor by stacking on the last dimension
        pred = torch.stack([out_states["i"], out_states["v"]], dim=-1)  # shape (B, T, 2)
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

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
from ..buck_converter_classes import BuckConverterDynamics, BUCK_PARAM_RESCALING

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
        
        self.param_rescaling = self._define_parameter_rescaling()
        self._initialize_log_parameters(param_init)
        self.system_dynamics: SystemDynamics = self._define_system_dynamics()

    def _initialize_log_parameters(self, param_init: Parameters):
        """
        Create learnable *log-space* parameters from an arbitrary Parameters instance.
        - Scalars -> nn.Parameter
        - Python list/tuple -> nn.ParameterList (one Parameter per element)
        - torch.Tensor with ndim>=1 -> nn.ParameterList (split along dim 0)
        """
        log_params = make_log_param(param_init, rescaling=self.param_rescaling)

        # keep track of what we created so we can rebuild Parameters later
        self._scalar_keys: list[str] = []
        self._list_keys: list[str] = []

        for key, val in log_params.params.items():
            # Case 1: Python list/tuple of scalars/tensors
            if isinstance(val, (list, tuple)):
                plist = nn.ParameterList(
                    [nn.Parameter(torch.as_tensor(v, dtype=torch.float32)) for v in val]
                )
                setattr(self, f"log__{key}", plist)
                self._list_keys.append(key)
                continue

            # Case 2: Tensor sequence (rank >= 1) -> split along the first dimension
            if isinstance(val, torch.Tensor) and val.ndim >= 1:
                # Do NOT call .unbind() on a leaf with grad; wrap into Parameters first
                elems = [t for t in val]  # iterates dim-0 slices
                plist = nn.ParameterList(
                    [nn.Parameter(torch.as_tensor(e, dtype=torch.float32)) for e in elems]
                )
                setattr(self, f"log__{key}", plist)
                self._list_keys.append(key)
                continue

            # Case 3: Scalar (float/int/0-D tensor/nn.Parameter)
            p = nn.Parameter(torch.as_tensor(val, dtype=torch.float32))
            setattr(self, f"log__{key}", p)
            self._scalar_keys.append(key)


    @property
    def logparams(self) -> Parameters:
        """
        Return a Parameters holding the *current* log-space values straight from the module.
        (Preserves tensors/nn.Parameters; no casting to float.)
        """
        data = {}
        for k in self._scalar_keys:
            data[k] = getattr(self, f"log__{k}")
        for k in self._list_keys:
            plist: nn.ParameterList = getattr(self, f"log__{k}")
            data[k] = [p for p in plist]
        return Parameters(**data)

    def get_estimates(self) -> Parameters:
        """Return current parameters in physical units (and inverse scaling)."""
        return reverse_log_param(self.logparams, self.param_rescaling)

    def _logparam_name_map(self) -> list[tuple[str, str]]:
        """
        Map flat display names (e.g., 'Rloads1') to module attribute paths:
        ('L', 'log__L'), ('Rloads1', 'log__Rloads.0'), ...
        Works for any list/scalar keys.
        """
        mapping: list[tuple[str, str]] = []
        for disp, _ in self.logparams.iterator():
            m = re.match(r"^(.*?)(\d+)$", disp)
            if m:
                base, idx = m.group(1), int(m.group(2)) - 1
                mapping.append((disp, f"log__{base}.{idx}"))
            else:
                mapping.append((disp, f"log__{disp}"))
        return mapping
    # def _physical(self) -> Parameters:
    #     """Return current parameters in physical units (inverse scaling)."""
    #     return reverse_log_param(self.logparams)

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
    
    def _define_parameter_rescaling(self) -> Optional[Parameters | None]:
        """Define the parameter rescaling factors."""
        return None

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
    def _define_parameter_rescaling(self) -> Optional[Parameters | None]:
        return None
    
    def _define_system_dynamics(self) -> SystemDynamics:
        return BuckConverterDynamics()

    def forward(self, X: torch.Tensor):
        """
        X: [batch, n_transients, 4] -> (i_n, v_n, D, dt)

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
        pred: Forward prediction of the buck converter states at time n+1. Shape (B, T, 2)
        """

        i, v = X[..., 0], X[..., 1]
        D, dt = X[..., 2], X[..., 3]

        input_states = States({"i": i[:-1], "v": v[:-1]})
        controls = {"D": D[:-1]}
        time_steps = dt[:-1]
        
        params = self.get_estimates().expand_torch_sequences()

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
    
    def _define_system_dynamics(self) -> SystemDynamics:
        return BuckConverterDynamics()

    def forward(self, X: torch.Tensor):
        """
        X: [batch, n_transients, 4] -> (i_n, v_n, D, dt)

        Returns:
        -------
        torch.tensor (2, B, T, 2)

        along the first axis of the tensor:
        fwd_pred: Forward prediction of the buck converter states at time n+1. Shape (B, T, 2)
        bck_pred: Backward prediction of the buck converter states at time n-1. Shape (B, T, 2)
        """

        i, v = X[..., 0], X[..., 1]
        D, dt = X[..., 2], X[..., 3]

        input_states_fwd = States({"i": i[:-1], "v": v[:-1]})
        input_states_bck = States({"i": i[1:], "v": v[1:]})
        controls = {"D": D[:-1]}
        time_steps = dt[:-1]
        
        params = self.get_estimates().expand_torch_sequences()

        # Forward and backward predictions (vectorized)
        out_states_fwd = self._rk4_step(
            time_steps=time_steps, 
            states=input_states_fwd, 
            params=params, 
            controls=controls, 
            sign=+1
            )

        out_states_bck = self._rk4_step(
            time_steps=time_steps, 
            states=input_states_bck, 
            params=params, 
            controls=controls, 
            sign=-1
            )
        
        pred_fwd = torch.stack([out_states_fwd["i"], out_states_fwd["v"]], dim=-1)  # shape (B, T, 2)
        pred_bck = torch.stack([out_states_bck["i"], out_states_bck["v"]], dim=-1)  # shape (B, T, 2)
        return torch.stack((pred_fwd, pred_bck), dim=0)  # shape (2, B, T, 2)

        # # transform the tuple into a tensor by stacking on the last dimension
        # pred = torch.stack([out_states["i"], out_states["v"]], dim=-1)  # shape (B, T, 2)


        # p = self.get_estimates().expand_torch_sequences()

        # # Forward and backward predictions (vectorized)
        # fwd_pred = self._rk4_step(i[:-1], v[:-1], D[:-1], dt[:-1], p, sign=+1)
        # bck_pred = self._rk4_step(i[1:], v[1:], D[:-1], dt[:-1], p, sign=-1)
        # # note that in the bck_pred we use D[:-1] and dt[:-1], this is because we want to predict the
        # # voltages and currents at the time n, given their values after the dt time at time n+1. However,
        # # D and dt refer to the time step from n to n+1, so we need to use the same D and dt as in the forward prediction.

        # # transform the tuple into a tensor by stacking on the last dimension
        # fwd_pred = torch.stack(fwd_pred, dim=-1)  # shape (B, T, 2)
        # bck_pred = torch.stack(bck_pred, dim=-1)  # shape (B, T, 2)

        # return torch.stack((fwd_pred, bck_pred), dim=0)  # shape (2, B, T, 2)

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

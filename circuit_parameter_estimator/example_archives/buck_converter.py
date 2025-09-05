"""
examples_archive/buck_baseline.py

Self-contained example “design file” that:
  • Defines buck-specific parameter container + units
  • Implements continuous-time dynamics for (i, v)
  • Provides ready-to-use parameter archives (TRUE/NOMINAL/…)
  • Exposes two estimators: forward-only and forward+backward

NOTES
-----
- Files like this one live in `examples_archive/` so each hardware setup or
  experiment can declare its own parameters/dynamics/estimator *without* touching
  library code. Treat this as a “manifest” for one design.
- Keep the public surface minimal: params, dynamics, estimator(s), constants.
"""

from __future__ import annotations

from typing import Dict, Final, Mapping, Optional, Tuple
from types import MappingProxyType

import torch

from circuit_parameter_estimator.model.system_dynamics import SystemDynamics
from circuit_parameter_estimator.model.system_state import States
from circuit_parameter_estimator.model.model_base import BaseBuckEstimator
from circuit_parameter_estimator.parameters.parameter_class import (
    Parameters,
    Units,
    Scalar,
    Seq,
)

# ---------------------------------------------------------------------------
# Measurement labels
# ---------------------------------------------------------------------------

MeasurementLabel = str
MeasurementNumber = int
MeasurementGroup = Mapping[MeasurementNumber, MeasurementLabel]


class MeasurementGroupArchive:
    """
    Groups all measurement numbering dictionaries in a read-only, type-safe way.

    NOTE
    ----
    - Using MappingProxyType makes these dicts immutable at runtime.
    - Add new groups as new class attributes, e.g. `MY_LAB_2025 = MappingProxyType({...})`.
    """

    # Shuai's measurement numbering
    SHUAI_ORIGINAL: Final[MeasurementGroup] = MappingProxyType(
        {
            0: "ideal",
            1: "ADC_error",
            2: "Sync Error",
            3: "5 noise",
            4: "10 noise",
            5: "ADC-Sync-5noise",
            6: "ADC-Sync-10noise",
        }
    )


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


class BuckConverterParams(Parameters):
    """
    Physical parameters for a buck converter.

    Fields
    ------
    L      : Inductance (H)
    RL     : Inductor series resistance (Ω)
    C      : Capacitance (F)
    RC     : Capacitor ESR (Ω)
    Rdson  : Switch on-resistance (Ω)
    Rloads : Load resistance(s) (Ω) — can be a list/tensor for multi-load datasets
    Vin    : Input voltage (V)
    VF     : Diode forward drop (V)

    NOTE
    ----
    - This subclasses the project’s `Parameters` so it integrates with
      logging, transforms (e.g., log-params), and broadcasting via
      `expand_torch_sequences()`.
    """

    L: Scalar
    RL: Scalar
    C: Scalar
    RC: Scalar
    Rdson: Scalar
    Rloads: Seq  # e.g., [Rload1, Rload2, Rload3] or a tensor
    Vin: Scalar
    VF: Scalar

    def __init__(
        self,
        L: Scalar,
        RL: Scalar,
        C: Scalar,
        RC: Scalar,
        Rdson: Scalar,
        Rloads: Seq,
        Vin: Scalar,
        VF: Scalar,
    ):
        super().__init__(
            L=L,
            RL=RL,
            C=C,
            RC=RC,
            Rdson=Rdson,
            Rloads=Rloads,
            Vin=Vin,
            VF=VF,
        )

        # Attach display/eng units (used by plotting/formatters elsewhere)
        units = Units(
            L="H",
            RL="Ω",
            C="F",
            RC="Ω",
            Rdson="Ω",
            Rloads="Ω",
            Vin="V",
            VF="V",
        )
        self._assign_unit_class(units)


# ---------------------------------------------------------------------------
# Continuous-time dynamics (discretized by the model via RK4)
# ---------------------------------------------------------------------------


class BuckConverterDynamics(SystemDynamics):
    """
    Continuous-time state dynamics for a buck converter.

    States
    ------
    i : Inductor current
    v : Capacitor/load voltage

    Controls
    --------
    D : Duty cycle (0..1)

    Params
    ------
    p : BuckConverterParams

    NOTE
    ----
    - We implement small helpers `_di` and `_dv` for clarity and reuse.
    """

    @staticmethod
    def _di(
        i_k: torch.Tensor, v_k: torch.Tensor, D: torch.Tensor, p: BuckConverterParams
    ) -> torch.Tensor:
        """
        di/dt at time k.
        S is the instantaneous switch state (here we pass the duty `D` continuously;
        for averaged modeling, S ≈ D).
        """
        # -( (D*Rdson + RL)*i + v - D*Vin + (1 - D)*VF ) / L
        return -((D * p.Rdson + p.RL) * i_k + v_k - D * p.Vin + (1 - D) * p.VF) / p.L

    @staticmethod
    def _dv(
        i_k: torch.Tensor,
        v_k: torch.Tensor,
        p: BuckConverterParams,
        rload: torch.Tensor,
        di: torch.Tensor,
    ) -> torch.Tensor:
        """
        dv/dt at time k.
        """
        # (C*RC*r * di + r*i - v) / (C*(RC + r))
        return (p.C * p.RC * rload * di + rload * i_k - v_k) / (p.C * (p.RC + rload))

    def dynamics(
        self,
        current_states: States,
        params: BuckConverterParams,
        controls: Dict[str, torch.Tensor],
    ) -> States:
        """
        Compute (di/dt, dv/dt) for the given state and control.

        Parameters
        ----------
        current_states : States with keys {"i", "v"}
        params         : BuckConverterParams
        controls       : {"D": duty_cycle}

        Returns
        -------
        States({"i": di, "v": dv})
        """
        i = current_states["i"]
        v = current_states["v"]
        D = controls["D"]

        # Single-pass computation (avoids recomputing di in _dv)
        di = self._di(i_k=i, v_k=v, D=D, p=params)

        # NOTE: Rloads can be vectorized; broadcasting handled by upstream call to
        # `params.expand_torch_sequences()` inside the estimator.
        dv = self._dv(i_k=i, v_k=v, p=params, rload=params.Rloads, di=di)

        return States({"i": di, "v": dv})


# ---------------------------------------------------------------------------
# Parameter archives (TRUE / NOMINAL / REL_TOL / RESCALING)
# ---------------------------------------------------------------------------


class ParameterArchive:
    """
    Physical and nominal parameter values used in the project.

    NOTES
    -----
    - TRUE     : ground-truth (for sims or validation datasets)
    - NOMINAL  : datasheet/initial guesses
    - REL_TOL  : relative tolerances (used to build priors/noise scales)
    - RESCALING: optional per-parameter multipliers to improve optimizer numerics
                 (kept here for completeness; estimators may ignore it)
    """

    TRUE: Final = BuckConverterParams(
        L=7.25e-4,
        RL=0.314,
        C=1.645e-4,
        RC=0.201,
        Rdson=0.221,
        Rloads=[3.1, 10.2, 6.1],
        Vin=48.0,
        VF=1.0,
    )

    NOMINAL: Final = BuckConverterParams(
        L=6.8e-4,
        RL=0.4,
        C=1.5e-4,
        RC=0.25,
        Rdson=0.25,
        Rloads=[3.3, 10.0, 6.8],
        Vin=46.0,
        VF=1.1,
    )

    REL_TOL: Final = BuckConverterParams(
        L=0.50,
        RL=0.40,
        C=0.50,
        RC=0.50,
        Rdson=0.50,
        Rloads=[0.30, 0.30, 0.30],
        Vin=0.30,
        VF=0.30,
    )

    RESCALING: Final = BuckConverterParams(
        L=1e4,
        RL=1e1,
        C=1e4,
        RC=1e1,
        Rdson=1e1,
        Rloads=[1.0, 1.0, 1.0],
        Vin=1e-1,
        VF=1.0,
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


class BuckParamEstimator(BaseBuckEstimator):
    """
    Physics-informed NN for parameter estimation in a buck converter.
    Forward prediction only (n -> n+1).

    Input X
    -------
    X: Tensor with shape [N, T, 4] where columns are (i_n, v_n, D_n, dt_n).

    Output
    ------
    pred: Tensor with shape [N-1, T, 2] for (i_{n+1}, v_{n+1}).

    NOTES
    -----
    - The integrator call `_rk4_step` expects:
        time_steps, states={'i','v'}, params, controls={'D'}, sign=+1
    - `get_estimates().expand_torch_sequences()` broadcasts scalar params
      across (N-1, T) so vector loads/different transients are supported.
    """

    def _define_parameter_rescaling(self) -> Optional[Parameters]:
        # NOTE: If you want numeric rescaling, return ParameterArchive.RESCALING.
        return None

    def _define_system_dynamics(self) -> SystemDynamics:
        return BuckConverterDynamics()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Slice inputs
        i, v = X[..., 0], X[..., 1]
        D, dt = X[..., 2], X[..., 3]

        # Prepare states/controls/time for steps from n -> n+1 (exclude last row)
        input_states = States({"i": i[:-1], "v": v[:-1]})
        controls = {"D": D[:-1]}
        time_steps = dt[:-1]

        # Broadcast parameters to match batch/time shapes
        params = self.get_estimates().expand_torch_sequences()

        # Single RK4 step (vectorized over [N-1, T])
        out_states = self._rk4_step(
            time_steps=time_steps,
            states=input_states,
            params=params,
            controls=controls,
            sign=+1,
        )

        # Stack to (N-1, T, 2)
        pred = torch.stack([out_states["i"], out_states["v"]], dim=-1)
        return pred

    def targets(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Targets aligned with the forward pass: the next-step states.
        Returns shape [N-1, T, 2] matching `forward`.
        """
        return X[1:, :, :2].clone().detach()


class BuckParamEstimatorFwdBck(BaseBuckEstimator):
    """
    Physics-informed NN for parameter estimation in a buck converter.
    Produces *both* forward (n -> n+1) and backward (n -> n-1) predictions.

    Input X
    -------
    X: Tensor with shape [N, T, 4] where columns are (i_n, v_n, D_n, dt_n).

    Output
    ------
    pred: Tensor with shape [2, N-1, T, 2]
          pred[0] = forward (i_{n+1}, v_{n+1})
          pred[1] = backward(i_{n-1}, v_{n-1})

    NOTES
    -----
    - Backward integration uses the *same* (D_n, dt_n) that advanced the system
      from n -> n+1. Conceptually, we integrate with sign = -1 to “undo” it.
    """

    def _define_system_dynamics(self) -> SystemDynamics:
        return BuckConverterDynamics()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        i, v = X[..., 0], X[..., 1]
        D, dt = X[..., 2], X[..., 3]

        # Forward: states at n predict n+1 using (D_n, dt_n)
        input_states_fwd = States({"i": i[:-1], "v": v[:-1]})
        # Backward: states at n+1 predict n using the same (D_n, dt_n), sign = -1
        input_states_bck = States({"i": i[1:], "v": v[1:]})

        controls = {"D": D[:-1]}
        time_steps = dt[:-1]

        params = self.get_estimates().expand_torch_sequences()

        out_states_fwd = self._rk4_step(
            time_steps=time_steps,
            states=input_states_fwd,
            params=params,
            controls=controls,
            sign=+1,
        )

        out_states_bck = self._rk4_step(
            time_steps=time_steps,
            states=input_states_bck,
            params=params,
            controls=controls,
            sign=-1,
        )

        pred_fwd = torch.stack([out_states_fwd["i"], out_states_fwd["v"]], dim=-1)
        pred_bck = torch.stack([out_states_bck["i"], out_states_bck["v"]], dim=-1)
        return torch.stack((pred_fwd, pred_bck), dim=0)  # (2, N-1, T, 2)

    def targets(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Targets for both directions:
          - forward target : x_{n+1}
          - backward target: x_{n}
        Returns shape [2, N-1, T, 2]
        """
        target_fwd = X[1:, :, :2].clone().detach()
        target_bck = X[:-1, :, :2].clone().detach()
        return torch.stack((target_fwd, target_bck), dim=0)


# ---------------------------------------------------------------------------
# Public module surface 
# ---------------------------------------------------------------------------

__all__ = [
    "MeasurementGroupArchive",
    "BuckConverterParams",
    "BuckConverterDynamics",
    "ParameterArchive",
    "BuckParamEstimator",
    "BuckParamEstimatorFwdBck",
]

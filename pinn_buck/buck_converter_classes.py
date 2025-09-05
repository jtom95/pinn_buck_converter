from .model.system_dynamics import SystemDynamics
from .model.system_state_class import States
from .parameters.parameter_class import Parameters, Units, Scalar, Seq
from typing import List, Dict, Final
import torch


class BuckConverterParams(Parameters):
    L: Scalar
    RL: Scalar
    C: Scalar
    RC: Scalar
    Rdson: Scalar
    Rloads: Seq  # Rload1, Rload2, Rload3
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


class BuckConverterDynamics(SystemDynamics):
    """
    Dynamics for a buck converter.
    States keys: 'i', 'v', 'D', ['dt' (default)]
    Params:
    """

    @staticmethod
    def _di(i_k, v_k, S, p: BuckConverterParams):
        return -((S * p.Rdson + p.RL) * i_k + v_k - S * p.Vin + (1 - S) * p.VF) / p.L

    @staticmethod
    def _dv(i_k, v_k, S, p: BuckConverterParams, rload, di):
        return (p.C * p.RC * rload * di + rload * i_k - v_k) / (p.C * (p.RC + rload))

    def dynamics(self, current_states: States, params: BuckConverterParams, controls: Dict[str, torch.Tensor]) -> States:
        i = current_states["i"]
        v = current_states["v"]
        D = controls["D"]

        di = -((D * params.Rdson + params.RL) * i + v - D * params.Vin + (1 - D) * params.VF) / params.L
        dv = (params.C * params.RC * params.Rloads * di + params.Rloads * i - v) / (
            params.C * (params.RC + params.Rloads)
        )

        # return the differentials of the state
        return States({"i": di, "v": dv})


class ParameterArchive:
    """Physical and nominal parameter values used in the project."""

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
        RL=0.4,
        C=0.50,
        RC=0.50,
        Rdson=0.5,
        Rloads=[0.3, 0.3, 0.3],
        Vin=0.3,
        VF=0.3,
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

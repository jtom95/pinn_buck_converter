from .model.system_dynamics import SystemDynamics
from .model.system_state_class import States
from .parameters.parameter_class import Parameters, Scalar, Seq
from typing import List, Dict
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


class BuckConverterDynamics(SystemDynamics):
    """
    Dynamics for a buck converter.
    State keys: 'i', 'v', 'D', ['dt' (default)]
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

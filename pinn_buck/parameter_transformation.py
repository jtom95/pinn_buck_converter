import torch
from .parameters.parameter_class import Parameters
from .constants import ParameterConstants 


import torch
from typing import Optional, Union


def make_log_param(params: Parameters) -> Parameters:
    """
    Convert physical parameters into log-space Parameters,
    applying scaling factors from ParameterConstants.SCALE first.

    Each output entry is a torch.Tensor (or nn.Parameter) that stays in
    the autograd graph.
    """

    def _to_log(
        value: Union[float, torch.Tensor], scale: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        v = torch.as_tensor(value, dtype=torch.float32)
        s = torch.as_tensor(scale, dtype=torch.float32)
        return torch.log(v * s)

    # safer: align by names, not just zip order
    scales = dict(ParameterConstants.SCALE.iterator())  # e.g. {"L": sL, "RL": sRL, ...}

    scaled_items = []
    for name, value in params.iterator():
        scale = scales.get(name, 1.0)
        scaled_items.append((name, _to_log(value, scale)))

    return Parameters.build_from_flat(scaled_items)


def reverse_log_param(log_param: Parameters) -> Parameters:
    """
    Convert log-space parameters (scalars or tensors/nn.Parameters)
    back to physical units using: phys = exp(log)/scale.
    """
    scales = dict(ParameterConstants.SCALE.iterator())  # {"L": sL, "RL": sRL, "Rload1": sR1, ...}
    items = []
    for name, log_value in log_param.iterator():
        s = torch.as_tensor(scales.get(name, 1.0), dtype=torch.float32)
        lv = torch.as_tensor(log_value, dtype=torch.float32)
        items.append((name, torch.exp(lv) / s))
    return Parameters.build_from_flat(items)

import torch
from .config import Parameters
from .constants import ParameterConstants 


import torch
from typing import Optional


def make_log_param(params: Parameters) -> Parameters:
    """
    Convert physical parameters into log-space trainable Parameters,
    applying the scaling factors from ParameterConstants.SCALE first.
    """

    def _to_log(value: float) -> torch.Tensor:
        return torch.log(torch.as_tensor(value, dtype=torch.float32))

    # Multiply by SCALE for each parameter in iterator order
    scaled_items = []
    for (name, value), (_, scale) in zip(params.iterator(), ParameterConstants.SCALE.iterator()):
        scaled_items.append((name, _to_log(value * scale)))

    return Parameters.build_from_all_names_iterator(scaled_items)


def reverse_log_param(log_param: Parameters) -> Parameters:
    """
    Convert log-space parameters back to physical units,
    applying inverse scaling from ParameterConstants.SCALE.
    """

    def _to_phys(log_value: float, scale: float) -> torch.Tensor:
        return torch.exp(torch.as_tensor(log_value, dtype=torch.float32)) / scale

    unscaled_items = []
    for (name, log_value), (_, scale) in zip(
        log_param.iterator(), ParameterConstants.SCALE.iterator()
    ):
        unscaled_items.append((name, _to_phys(log_value, scale)))

    return Parameters.build_from_all_names_iterator(unscaled_items)

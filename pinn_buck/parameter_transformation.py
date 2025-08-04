import torch
from .config import Parameters, _SCALE


def make_log_param(params: Parameters) -> Parameters:
    """Convert physical parameters into log‑space trainable Parameters."""
    def _to_log(value: float) -> torch.Tensor:
        """Convert a physical value to log‑space."""
        return torch.log(torch.as_tensor(value, dtype=torch.float32))

    return Parameters(
        L=_to_log(params.L*_SCALE["L"]),
        RL=_to_log(params.RL*_SCALE["RL"]),
        C=_to_log(params.C*_SCALE["C"]),
        RC=_to_log(params.RC*_SCALE["RC"]),
        Rdson=_to_log(params.Rdson*_SCALE["Rdson"]),
        Rloads= [
            _to_log(rload * scale) for rload, scale in zip(params.Rloads, _SCALE["Rloads"])
        ],
        Vin=_to_log(params.Vin*_SCALE["Vin"]),
        VF=_to_log(params.VF*_SCALE["VF"])
    )


def reverse_log_param(log_param: Parameters) -> float:
    """Convert a log‑space parameter back to physical units."""
    return Parameters(
        L=torch.exp(torch.as_tensor(log_param.L, dtype=torch.float32)) / _SCALE["L"],
        RL=torch.exp(torch.as_tensor(log_param.RL, dtype=torch.float32)) / _SCALE["RL"],
        C=torch.exp(torch.as_tensor(log_param.C, dtype=torch.float32)) / _SCALE["C"],
        RC=torch.exp(torch.as_tensor(log_param.RC, dtype=torch.float32)) / _SCALE["RC"],
        Rdson=torch.exp(torch.as_tensor(log_param.Rdson, dtype=torch.float32)) / _SCALE["Rdson"],
        Rloads=[
            torch.exp(torch.as_tensor(rload, dtype=torch.float32)) / scale
            for rload, scale in zip(log_param.Rloads, _SCALE["Rloads"])
        ],
        Vin=torch.exp(torch.as_tensor(log_param.Vin, dtype=torch.float32)) / _SCALE["Vin"],
        VF=torch.exp(torch.as_tensor(log_param.VF, dtype=torch.float32)) / _SCALE["VF"],
    )

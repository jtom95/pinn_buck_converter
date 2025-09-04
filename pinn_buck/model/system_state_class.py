from typing import Dict, Mapping, Callable, Tuple, Any
import torch
from torch import nn


class State(dict):
    """
    Mapping[str, torch.Tensor] with vector-space ops for RK4.
    Math ops act on all keys EXCEPT 'dt' (which is carried through unchanged).
    """

    def __init__(self, data: Mapping[str, torch.Tensor] = (), dt: torch.Tensor = None):
        if "dt" in data: 
            raise ValueError("State 'dt' key must be provided as dt argument, not in data mapping")
        if dt is None: 
            raise ValueError("State 'dt' key must be provided as dt argument")
        self.dt = dt
        super().__init__({k: torch.as_tensor(v) for k, v in dict(data).items()})
    
    def _dtype(self):
        for k, v in self.items():
            return v.dtype
        return torch.float32

    def _apply_bin(
        self, other: "State", op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> "State":
        if not torch.allclose(self.dt, other.dt):
            raise ValueError("Cannot apply binary operation to States with different 'dt' values")
        
        # only apply to common keys
        k1 = set(self.keys())
        k2 = set(other.keys())

        kcommon = k1.intersection(k2)
        k1_only = k1 - kcommon
        k2_only = k2 - kcommon
        
        if len(kcommon) == 0:
            return k1_only.union(k2_only)
    
        common_out: Dict[str, torch.Tensor] = {}
        for k in kcommon:
            common_out[k] = op(self[k], other[k])
        # unite with non-common keys
        out = {**common_out, **{k: self[k] for k in k1_only}, **{k: other[k] for k in k2_only}}
        return State(out, dt=self.dt)
        

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "State":
        out = {k: fn(self[k]) for k in self.keys()}
        return State(out, self.dt)

    def __add__(self, other: "State") -> "State":
        return self._apply_bin(other, torch.add)

    def __radd__(self, other: "State") -> "State":
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, a: torch.Tensor | float) -> "State":
        a = torch.as_tensor(a, dtype=self._dtype(), device=self.dt.device)
        out = {k: self[k] * a for k in self.keys()}
        return State(out, self.dt)

    def __rmul__(self, a: torch.Tensor | float) -> "State":
        return self.__mul__(a)

    def __repr__(self):
        return f"State({super().__repr__()})"

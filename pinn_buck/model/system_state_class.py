from typing import Dict, Mapping, Callable, Tuple, Any
import torch
from torch import nn


class States(dict):
    """
    Mapping[str, torch.Tensor] with vector-space ops for RK4.
    Math ops act on all keys EXCEPT 'dt' (which is carried through unchanged).
    """

    def __init__(self, data: Mapping[str, torch.Tensor] = ()):
        super().__init__({k: torch.as_tensor(v) for k, v in dict(data).items()})
    
    def _dtype(self):
        for k, v in self.items():
            return v.dtype
        return torch.float32

    def _apply_bin(
        self, other: "States", op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> "States":
        
        # only apply to common keys
        k1 = set(self.keys())
        k2 = set(other.keys())

        kcommon = k1.intersection(k2)
        k1_only = k1 - kcommon
        k2_only = k2 - kcommon
        
        if len(kcommon) == 0:
            return States({k: self[k] for k in k1_only} | {k: other[k] for k in k2_only})

        common_out: Dict[str, torch.Tensor] = {}
        for k in kcommon:
            common_out[k] = op(self[k], other[k])
        # unite with non-common keys
        out = {**common_out, **{k: self[k] for k in k1_only}, **{k: other[k] for k in k2_only}}
        return States(out)
    
    def pop_key(self, name: str) -> "States":
        """Return a NEW State without `name` (if present)."""
        out = {k: v for k, v in self.items() if k != name}
        return States(out)

    def add_key(self, name: str, value: torch.Tensor) -> "States":
        """Return a NEW States with `name` added (or replaced)."""
        out = dict(self)
        out[name] = torch.as_tensor(value, dtype=next(iter(self.values())).dtype if len(self) else torch.float32,
                                    device=self.dt.device)
        return States(out)

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "States":
        out = {k: fn(self[k]) for k in self.keys()}
        return States(out)

    def __add__(self, other: "States") -> "States":
        return self._apply_bin(other, torch.add)

    def __radd__(self, other: "States") -> "States":
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, a: torch.Tensor | float) -> "States":
        a = torch.as_tensor(a, dtype=self._dtype(), device=next(iter(self.values())).device)
        out = {k: self[k] * a for k in self.keys()}
        return States(out)

    def __rmul__(self, a: torch.Tensor | float) -> "States":
        return self.__mul__(a)

    def __repr__(self):
        return f"States({super().__repr__()})"

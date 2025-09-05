import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Tuple,
    Union,
    Optional,
    Final,
    FrozenSet
)
from pathlib import Path
import json
import re

Scalar = Union[int, float, torch.Tensor, torch.nn.Parameter]
Seq = Union[List[Scalar], Tuple[Scalar, ...], torch.Tensor]
Value = Union[Scalar, Seq]



def _is_seq_like(x: Any) -> bool:
    # list/tuple of scalars/tensors
    return isinstance(x, (list, tuple))


def _is_tensor_sequence(x: Any) -> bool:
    # torch.Tensor with at least one dimension (rank>=1)
    return isinstance(x, torch.Tensor) and x.ndim >= 1


class Units:
    def __init__(self, **kwargs: str):
        self._units: Dict[str, str] = dict(kwargs)

    def __getitem__(self, key: str) -> Optional[str]:
        return self._units.get(key)

    def __setitem__(self, key: str, value: str) -> None:
        self._units[key] = value

    def __delitem__(self, key: str) -> None:
        del self._units[key]

    def items(self) -> Iterable[Tuple[str, str]]:
        return self._units.items()
    
    def get_from_iterator_name(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if name in self._units:
            return self._units[name]
        # try to strip trailing digits for flattened names
        m = re.match(r"^(.*?)(\d+)$", name)
        if m:
            base = m.group(1)
            return self._units.get(base, default)
        return default
    
    @classmethod
    def make_default(cls, parameters: "Parameters", default_unit: str = "") -> "Units":
        unit_dict = {}  
        for name, _ in parameters.params.items():
            unit_dict[name] = default_unit
        return cls(**unit_dict)
            

class Parameters:
    DEFAULT_FILE_PATTERN: Final = "*.params.json"
    _reserved_attrs = {
        "params", "_units", "DEFAULT_FILE_PATTERN", "_reserved_attrs",
        "iterator", "__len__", "expand", "get_all_values", "get_all_names",
        "get_from_iterator_name", "save", "load", "_to_jsonable",
        "from_mapping", "build_from_flat", "__getitem__", "__setitem__",
        "__delattr__", "__getattr__", "__setattr__", "__dir__", "pop",
    }
    params: Dict[str, Value]
    _frozen_keys: FrozenSet[str]

    def __init__(self, **kwargs: Value):
        # store params (keys frozen after init)
        super().__setattr__("params", dict(kwargs))
        # lock the set of keys
        super().__setattr__("_frozen_keys", frozenset(self.params.keys()))
        # assign the physical units (empty by default)
        self._assign_unit_class(Units())
        
    @property
    def units(self) -> Optional[Units]:
        return self._units
    
    def _assign_unit_class(self, units: Units) -> None:
        # check that all keys in units are valid parameter names
        for key in units._units.keys():
            if key not in self.params:
                raise ValueError(f"Unit key '{key}' not found in parameters.")
        # add the default unit for any missing keys
        for key in self.params.keys():
            if key not in units._units:
                units[key] = ""
        self._units = units

    # ========== iteration / flattening ==========
    def iterator(self) -> Iterator[Tuple[str, Scalar]]:
        """
        Yields (flat_name, value) pairs.
        - list/tuple -> enumerate: name1, name2, ...
        - tensor rank>=1 -> squeeze and enumerate to work with expanded tensors.
        - scalars (int/float/rank-0 tensor) -> name, value
        """
        for key, val in self.params.items():
            if _is_seq_like(val):
                for i, v in enumerate(val, start=1):
                    yield f"{key}{i}", v
            elif _is_tensor_sequence(val):
                # slice along first dimension while preserving rest of dims
                for i in range(val.squeeze().shape[0]):
                    yield f"{key}{i+1}", val[i]
            else:
                yield key, val

    def __len__(self) -> int:
        return sum(1 for _ in self.iterator())

    # ========== attribute <-> dict sync (keys frozen) ==========
    def __getattr__(self, name: str) -> Any:
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")
    
    def __repr__(self) -> str:
        parts = []
        for name, value in self.iterator(): 
            unit = self.units.get_from_iterator_name(name, "")
            parts.append(f"{name}: {value}<{unit}>" if unit else f"{name}: <{value}>")
        return " - ".join(parts)

    def __setattr__(self, name: str, value: Any) -> None:
        # allow normal setting for reserved/internal attrs
        if name in self._reserved_attrs or name.startswith("__") or name == "_frozen_keys":
            super().__setattr__(name, value)
            return
        # only allow updates to existing parameter keys
        if name in self.params:
            self.params[name] = value
            return
        # forbid adding new parameter keys as attributes
        raise AttributeError(
            f"Cannot add new parameter '{name}' after initialization; keys are frozen. "
            f"Existing keys: {sorted(self.params.keys())}"
        )

    def __delattr__(self, name: str) -> None:
        # never allow deleting parameter keys
        if name in self.params:
            raise AttributeError("Deleting parameters is not allowed; keys are frozen.")
        # allow deleting non-param attrs (rarely needed)
        super().__delattr__(name)

    def __dir__(self) -> List[str]:
        return sorted(set(list(self.__dict__.keys()) + list(self.params.keys()) + list(type(self).__dict__.keys())))

    # ========== mapping convenience (keys frozen) ==========
    def __getitem__(self, key: str) -> Value:
        return self.params[key]

    def __setitem__(self, key: str, value: Value) -> None:
        if key not in self.params:
            raise KeyError(
                f"Cannot add new parameter '{key}' after initialization; keys are frozen. "
                f"Existing keys: {sorted(self.params.keys())}"
            )
        self.params[key] = value

    def pop(self, key: str, *default) -> Value:
        # no deletions
        raise TypeError("pop() is disabled; parameter keys are frozen.")

    # ========== helpers ==========
    def expand(self) -> Dict[str, float]:
        return {k: v for k, v in self.iterator()}

    def get_all_values(self) -> List[float]:
        return [v for _, v in self.iterator()]

    def get_all_names(self) -> List[str]:
        return [k for k, _ in self.iterator()]

    def get_from_iterator_name(self, name: str, default="raise error") -> float:
        flat = self.expand()
        if name not in flat:
            if default != "raise error":
                return default
            raise ValueError(f"Parameter '{name}' not found.")
        return flat[name]

    def expand_torch_sequences(self) -> "Parameters":
        """
        Convert all sequence-like params into broadcastable tensors with one
        leading axis per sequence param, ordered by first appearance.

        Example:
            x1 = [a, b]         -> shape (1, 2)
            x2 = [c, d, e]      -> shape (1, 1, 3)
            Resulting arithmetic naturally broadcasts over (2, 3) grid.
        """
        seq_keys: List[str] = []
        for k, v in self.params.items():
            if _is_seq_like(v) or _is_tensor_sequence(v):
                seq_keys.append(k)

        n_seq = len(seq_keys)
        if n_seq == 0:
            return self  # nothing to expand

        new_params: Dict[str, Value] = dict(self.params)  # copy shallow

        for dim, key in enumerate(seq_keys):
            values = self.params[key]

            if _is_seq_like(values):
                base = torch.stack([torch.as_tensor(x) for x in values])  # (L,)
            elif _is_tensor_sequence(values):
                base = values  # keep as tensor; preserve grad
            else:
                # scalar-like: nothing to do
                continue

            # base shape = (L, *rest)
            if base.ndim == 0:
                # degenerately scalar — treat like length-1 sequence
                base = base.view(1)

            L = base.shape[0]
            rest = list(base.shape[1:])

            # Build leading broadcast axes for all sequence params
            lead = [1] + [1] * n_seq
            lead[1 + dim] = L
            reshaped = base.reshape(*lead, *rest)

            new_params[key] = reshaped

        out = type(self).build_from_flat(new_params)
        out._assign_unit_class(self.units)  # copy units
        return out

    # ========== (de)serialization ==========
    def save(self, path: Union[str, Path]) -> None:
        file_suffix = self.DEFAULT_FILE_PATTERN[1:]  # ".params.json"
        path = Path(path)
        if not path.name.endswith(file_suffix):
            if path.suffix == ".json":
                path = path.with_name(path.stem + file_suffix)
            else:
                path = path.with_suffix(file_suffix)
        data = self._to_jsonable(self.params)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Parameters":
        path = Path(path)
        expected = cls.DEFAULT_FILE_PATTERN[1:]
        if not path.name.endswith(expected):
            raise Warning(f"Invalid file pattern: {path.name}. Expected pattern: {expected}")
        data = json.loads(path.read_text())
        return cls(**data)

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if isinstance(obj, (int, float, str)):
            return obj
        if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
            # scalar tensor?
            if obj.ndim == 0:
                return float(obj.detach().cpu().item())
            # non-scalar -> list
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [Parameters._to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: Parameters._to_jsonable(v) for k, v in obj.items()}
        return obj

    # ========== builders ==========
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Value]) -> "Parameters":
        return cls(**dict(mapping))

    @classmethod
    def build_from_flat(
        cls,
        flat: Union[Mapping[str, Scalar], Iterable[Tuple[str, Scalar]]],
        *,
        list_key_policy: Optional[Mapping[str, str]] = None,
    ) -> "Parameters":
        import re
        items = list(flat.items()) if isinstance(flat, Mapping) else list(flat)
        scalars: Dict[str, Scalar] = {}
        grouped: Dict[str, List[Tuple[int, Scalar]]] = {}
        rx = re.compile(r"^(.*?)(\d+)$")
        for name, value in items:
            m = rx.match(name)
            if m:
                base, idx = m.group(1), int(m.group(2))
                grouped.setdefault(base, []).append((idx, value))
            else:
                scalars[name] = value
        params: Dict[str, Value] = dict(scalars)
        list_key_policy = dict(list_key_policy or {})
        for base, pairs in grouped.items():
            pairs.sort(key=lambda t: t[0])
            _, values = zip(*pairs)
            container_key = list_key_policy.get(base, base)
            params[container_key] = list(values)
        return cls(**params)

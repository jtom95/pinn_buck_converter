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
Seq = Union[List[Scalar], Tuple[Scalar, ...]]
Value = Union[Scalar, Seq]


def _is_sequence(x: Any) -> bool:
    if isinstance(x, (str, bytes)):
        return False
    return isinstance(x, (list, tuple))


class Parameters:
    DEFAULT_FILE_PATTERN: Final = "*.params.json"
    _reserved_attrs = {
        "params", "DEFAULT_FILE_PATTERN", "_reserved_attrs",
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

    # ========== iteration / flattening ==========
    def iterator(self) -> Iterator[Tuple[str, Scalar]]:
        for key, val in self.params.items():
            if isinstance(val, (list, tuple)):
                for i, v in enumerate(val, start=1):
                    yield f"{key}{i}", v  # keep as tensor/Parameter/float intact
            else:
                yield key, val

    def __len__(self) -> int:
        return sum(1 for _ in self.iterator())

    # ========== attribute <-> dict sync (keys frozen) ==========
    def __getattr__(self, name: str) -> Any:
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

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

    def get_from_iterator_name(self, name: str) -> float:
        flat = self.expand()
        if name not in flat:
            raise ValueError(f"Parameter '{name}' not found.")
        return flat[name]

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

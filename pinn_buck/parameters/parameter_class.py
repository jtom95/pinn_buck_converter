from typing import List, NamedTuple, Mapping, Union, Iterable, Tuple, Final
import pandas as pd
import numpy as np
from pathlib import Path
import json

DEFAULT_FILE_PATTERN: Final = "*.params.json"

class Parameters(NamedTuple):
    L: float
    RL: float
    C: float
    RC: float
    Rdson: float
    Rloads: List[float] 
    Vin: float
    VF: float

    def iterator(self) -> iter:
        # Function to iterate through the Parameters returning parameter_name, parameter_value.
        # The tricky part is that Rloads is a list and the iterator should pass each of the values
        # separately, with name: Rload1, Rload2, etc.
        for name, value in self._get_params().items():
            if name == "Rloads":
                for i, load in enumerate(value):
                    yield f"Rload{i+1}", load
            else:
                yield name, value

    def __len__(self):
        # len function returns the number of total parameters we have when iterating, not the total number of fields
        return sum(1 for _ in self.iterator())

    def _get_params(self):
        return {
            "L": self.L,
            "RL": self.RL,
            "C": self.C,
            "RC": self.RC,
            "Rdson": self.Rdson,
            "Rloads": self.Rloads,
            "Vin": self.Vin,
            "VF": self.VF,
        }

    def get_from_iterator_name(self, param_name: str) -> float:
        names = self.get_all_names()
        # check if the param_name is present in the names
        if param_name in names:
            return self.get_all_values()[names.index(param_name)]
        raise ValueError(f"Parameter '{param_name}' not found.")

    def get_all_values(self) -> List[float]:
        return [value for _, value in self.iterator()]

    @classmethod
    def get_all_names(cls) -> List[str]:
        return [name for name, _ in cls.build_empty().iterator()]

    @classmethod
    def build_empty(cls) -> "Parameters":
        return cls(
            L=0.0,
            RL=0.0,
            C=0.0,
            RC=0.0,
            Rdson=0.0,
            Rloads=[0.0, 0.0, 0.0],
            Vin=0.0,
            VF=0.0,
        )

    @classmethod
    def build_from_all_names_iterator(
        cls, iterator: Union[Mapping[str, float], Iterable[Tuple[str, float]]]
    ) -> "Parameters":
        """
        Build a Parameters instance from an iterator or dict
        mapping 'L', 'RL', ..., 'Rload1', 'Rload2', ... to values.
        """
        if isinstance(iterator, Mapping):
            items = list(iterator.items())
        else:
            items = list(iterator)

        # Separate Rloads and scalar parameters
        rloads = []
        param_dict = {}
        for name, value in items:
            if name.startswith("Rload"):
                rloads.append((int(name[5:]), value))  # store index + value
            else:
                param_dict[name] = value

        # Sort Rloads by index so Rload1, Rload2, ... are in order
        rloads_sorted = [v for _, v in sorted(rloads, key=lambda t: t[0])]

        return cls(
            L=param_dict["L"],
            RL=param_dict["RL"],
            C=param_dict["C"],
            RC=param_dict["RC"],
            Rdson=param_dict["Rdson"],
            Rloads=rloads_sorted,
            Vin=param_dict["Vin"],
            VF=param_dict["VF"],
        )

    @classmethod
    def build_from_field_iterator(cls, m: Union[Mapping[str, float], Iterable[Tuple[str, float]]]) -> "Parameters":
        names = []
        values = []
        for name, value in m.items():
            names.append(name)
            values.append(value)
        return cls(*values)

    def save(self, path: Union[str, Path]):
        """
        Save Parameters instance to JSON file.
        """
        path = Path(path)
        file_pattern = DEFAULT_FILE_PATTERN

        # if it doesn't end in .laplace.json add it. If it just finishes in .json make it .laplace.json
        if not path.name.endswith(file_pattern):
            if path.name.endswith(".json"):
                path = path.with_name(path.stem + file_pattern[1:])
            else:
                path = path.with_suffix(file_pattern[1:])

        data = dict(self._get_params())  # includes Rloads list
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Parameters":
        """
        Load Parameters instance from JSON file.
        """
        path = Path(path)
        
        if not path.name.endswith(DEFAULT_FILE_PATTERN[1:]):
            raise Warning(
                f"Invalid file pattern: {path.name}. Expected pattern: {DEFAULT_FILE_PATTERN}"
            )
        
        with path.open("r") as f:
            data = json.load(f)

        return cls(
            L=data["L"],
            RL=data["RL"],
            C=data["C"],
            RC=data["RC"],
            Rdson=data["Rdson"],
            Rloads=data["Rloads"],
            Vin=data["Vin"],
            VF=data["VF"],
        )

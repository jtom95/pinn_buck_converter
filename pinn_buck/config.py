from typing import List, NamedTuple
import pandas as pd
import numpy as np
from pathlib import Path


# class Parameters(NamedTuple):
#     L: float
#     RL: float
#     C: float
#     RC: float
#     Rdson: float
#     Rload1: float
#     Rload2: float
#     Rload3: float
#     Vin: float
#     VF: float


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


# Nominal component values (physical units)
TRUE = Parameters(
    L=7.25e-4,
    RL=0.314,
    C=1.645e-4,
    RC=0.201,
    Rdson=0.221,
    Rloads= [3.1, 10.2, 6.1],  # Rload1, Rload2, Rload3
    Vin=48.0,
    VF=1.0,
)

# Initial physical guesses
INITIAL_GUESS = Parameters(
    L=2.0e-4,
    RL=0.0039,
    C=0.412e-4,
    RC=0.159,
    Rdson=0.122,
    Rloads = [1.22, 1.22, 1.22],  # Rload1, Rload2, Rload3
    Vin=8.7,
    VF=0.1,
)


# Scaling factors local to helpers – no need for global state
_SCALE = {
    "L": 1e4,
    "RL": 1e1,
    "C": 1e4,
    "RC": 1e1,
    "Rdson": 1e1,
    "Rloads": [1., 1., 1.],
    "Vin": 1e-1,  # inverse of 1e1 used previously
    "VF": 1.0,
}

__all__ = [
    "Parameters",
    "TRUE",
    "INITIAL_GUESS",
    "_SCALE",
]


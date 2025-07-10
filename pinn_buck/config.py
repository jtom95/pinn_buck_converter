from typing import List, NamedTuple
import pandas as pd
import numpy as np
from pathlib import Path


class Parameters(NamedTuple):
    L: float
    RL: float
    C: float
    RC: float
    Rdson: float
    Rload1: float
    Rload2: float
    Rload3: float
    Vin: float
    VF: float


# Nominal component values (physical units)
TRUE = Parameters(
    L=7.25e-4,
    RL=0.314,
    C=1.645e-4,
    RC=0.201,
    Rdson=0.221,
    Rload1=3.1,
    Rload2=10.2,
    Rload3=6.1,
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
    Rload1=1.22,
    Rload2=1.22,
    Rload3=1.22,
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
    "Rload1": 1.0,
    "Rload2": 1.0,
    "Rload3": 1.0,
    "Vin": 1e-1,  # inverse of 1e1 used previously
    "VF": 1.0,
}

__all__ = [
    "Parameters",
    "TRUE",
    "INITIAL_GUESS",
    "_SCALE",
]


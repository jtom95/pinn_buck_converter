from .config import Parameters
from typing import Mapping, Final
from types import MappingProxyType

## Parameter Constants
class ParameterConstants:
    """Physical and nominal parameter values used in the project."""

    TRUE : Final = Parameters(
        L=7.25e-4,
        RL=0.314,
        C=1.645e-4,
        RC=0.201,
        Rdson=0.221,
        Rloads=[3.1, 10.2, 6.1],
        Vin=48.0,
        VF=1.0,
    )

    NOMINAL: Final = Parameters(
        L=6.8e-4,
        RL=0.4,
        C=1.5e-4,
        RC=0.25,
        Rdson=0.25,
        Rloads=[3.3, 10.0, 6.8],
        Vin=46.0,
        VF=1.1,
    )

    REL_TOL: Final = Parameters(
        L=0.50,
        RL=0.4,
        C=0.50,
        RC=0.50,
        Rdson=0.5,
        Rloads=[0.3, 0.3, 0.3],
        Vin=0.3,
        VF=0.3,
    )

    SCALE: Final = Parameters(
        L = 1e4,
        RL = 1e1,
        C = 1e4,
        RC = 1e1,
        Rdson = 1e1,
        Rloads = [1.0, 1.0, 1.0],
        Vin = 1e-1,
        VF = 1.0,
    )

    # _SCALE: Final = MappingProxyType({
    #     "L": 1e4,
    #     "RL": 1e1,
    #     "C": 1e4,
    #     "RC": 1e1,
    #     "Rdson": 1e1,
    #     "Rloads": [1.0, 1.0, 1.0],
    #     "Vin": 1e-1,
    #     "VF": 1.0,
    # })

MeasurementLabel = str
MeasurementNumber = int
MeasurementGroup = Mapping[MeasurementNumber, MeasurementLabel]

class MeasurementGroupArchive:
    """Groups all measurement numbering dictionaries in a read-only, type-safe way."""

    # Shuai's measurement numbering
    SHUAI_ORIGINAL: Final[MeasurementGroup] = MappingProxyType(
        {
            0: "ideal",
            1: "ADC_error",
            2: "Sync Error",
            3: "5 noise",
            4: "10 noise",
            5: "ADC-Sync-5noise",
            6: "ADC-Sync-10noise",
        }
    )

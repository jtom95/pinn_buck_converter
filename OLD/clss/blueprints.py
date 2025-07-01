# Data Structure Class
from dataclasses import dataclass
import numpy as np

@dataclass
class DataStructure:
    x_current: np.ndarray
    y_current: np.ndarray
    x_volt: np.ndarray
    y_volt: np.ndarray
    d_switch: np.ndarray
    indicator: np.ndarray
    dt: np.ndarray

    @classmethod
    def from_dict(cls, data_dict: dict) -> "DataStructure":
        return cls(
            x_current=data_dict["CurrentInput"].squeeze(),
            y_current=data_dict["Current"].squeeze(),
            x_volt=data_dict["VoltageInput"].squeeze(),
            y_volt=data_dict["Voltage"].squeeze(),
            d_switch=data_dict["Dswitch"].squeeze(),
            indicator=data_dict["forwaredBackwaredIndicator"].squeeze(),
            dt=data_dict["dt"].squeeze(),
        )

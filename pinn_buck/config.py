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
NOMINAL = Parameters(
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


class TrainingRun:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def best_parameters(self) -> NamedTuple:
        """Get the best parameters from the DataFrame."""
        best_iter = self.df["loss"].idxmin()
        best_params = self.df.iloc[best_iter]
        return Parameters(
            L=best_params["L"],
            RL=best_params["RL"],
            C=best_params["C"],
            RC=best_params["RC"],
            Rdson=best_params["Rdson"],
            Rload1=best_params["Rload1"],
            Rload2=best_params["Rload2"],
            Rload3=best_params["Rload3"],
            Vin=best_params["Vin"],
            VF=best_params["VF"],
        )
    
    def drop_columns(self, columns: List[str]) -> "TrainingRun":
        """Drop specified columns from the DataFrame."""
        if isinstance(columns, str):
            columns = [columns]
        self.df.drop(columns=columns, inplace=True, errors='ignore')
        return self
        
    def save_to_csv(self, csv_path: Path):
        """Save the TrainingRun to a CSV file."""
        # check if file path is valid
        if not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.suffix == ".csv":
            raise ValueError(f"Expected a CSV file path, got {csv_path}")
        # save the DataFrame to CSV
        self.df.to_csv(csv_path, index=False)
    
    @classmethod
    def from_histories(cls, loss_history: List[float], param_history: List[Parameters], learning_rate: List[float]=None) -> "TrainingRun":
        """Create a TrainingRun from loss and parameter histories."""
        data = {
            "loss": loss_history,
            **{name: [getattr(p, name) for p in param_history] for name in Parameters._fields}
        }
        df = pd.DataFrame(data)
        if learning_rate is not None:
            df["learning_rate"] = learning_rate
        return TrainingRun(df)
        
    @classmethod
    def from_csv(cls, csv_path: str) -> "TrainingRun":
        """Load a TrainingRun from a CSV file."""
        df = pd.read_csv(csv_path)
        return TrainingRun(df)
        

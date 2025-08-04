from pathlib import Path
from typing import List, NamedTuple
import pandas as pd

from .config import Parameters


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
            Rloads = best_params["Rloads"],
            Vin=best_params["Vin"],
            VF=best_params["VF"],
        )

    def drop_columns(self, columns: List[str]) -> "TrainingRun":
        """Drop specified columns from the DataFrame."""
        if isinstance(columns, str):
            columns = [columns]
        self.df.drop(columns=columns, inplace=True, errors="ignore")
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
    def from_histories(
        cls,
        loss_history: List[float],
        param_history: List[Parameters],
        learning_rate: List[float] = None,
    ) -> "TrainingRun":
        """Create a TrainingRun from loss and parameter histories."""
        data = {
            "loss": loss_history,
            **{name: [getattr(p, name) for p in param_history] for name in Parameters._fields},
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

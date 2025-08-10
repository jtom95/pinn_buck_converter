from pathlib import Path
from typing import List, NamedTuple, Optional
import pandas as pd

from ..config import Parameters


class TrainingHistory:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_parameters(self, iter: int) -> Parameters:
        """Get parameters for a specific iteration."""
        if iter < 0 or iter >= len(self.df):
            raise IndexError(f"Iteration {iter} out of bounds for history with {len(self.df)} entries.")
        row = self.df.iloc[iter]
        return Parameters(
            L=row["L"],
            RL=row["RL"],
            C=row["C"],
            RC=row["RC"],
            Rdson=row["Rdson"],
            Rloads=row["Rloads"],
            Vin=row["Vin"],
            VF=row["VF"],
        )
        
    def get_best_idx(self, optimizer_type: Optional[str]=None) -> int:
        """Get the index of the best loss in the DataFrame."""
        # check if the optimizer type exists in the DataFrame
        if optimizer_type is None:
            return self.df["loss"].idxmin()
        
        if "optimizer" not in self.df.columns:
            raise ValueError("Optimizer type not found in the DataFrame.")
        if optimizer_type not in self.df["optimizer"].unique():
            raise ValueError(f"Optimizer type '{optimizer_type}' not found in the DataFrame. Optimizer types present are: {self.df['optimizer'].unique()}")
        return self.df[self.df["optimizer"] == optimizer_type]["loss"].idxmin()
    
    def get_best_parameters(self, optimizer_type: Optional[str]=None) -> Parameters:
        """Get the best parameters for a specific optimizer type."""
        best_idx = self.get_best_idx(optimizer_type)
        return self.get_parameters(best_idx)
    
    def get_best_loss(self, optimizer_type: Optional[str]=None) -> float:
        """Get the best loss value for a specific optimizer type."""
        best_idx = self.get_best_idx(optimizer_type)
        return self.df.at[best_idx, "loss"]
    
    def get_best_epoch(self, optimizer_type: Optional[str]=None) -> int:
        """Get the epoch corresponding to the best loss for a specific optimizer type."""
        best_idx = self.get_best_idx(optimizer_type)
        if "epoch" not in self.df.columns:
            raise ValueError("Epoch column not found in the DataFrame.")
        return self.df.at[best_idx, "epoch"]
    

    def drop_columns(self, columns: List[str]) -> "TrainingHistory":
        """Drop specified columns from the DataFrame."""
        if isinstance(columns, str):
            columns = [columns]
        self.df.drop(columns=columns, inplace=True, errors="ignore")
        return self

    def save_to_csv(self, csv_path: Path):
        """Save the TrainingHistory to a CSV file."""
        # check if file path is valid
        if not csv_path.parent.exists():
            raise FileNotFoundError(f"Output directory does not exist: {csv_path.parent}")
        if not csv_path.suffix == ".csv":
            raise Warning(f"File path {csv_path} does not end with .csv, saving as CSV anyway.")
        csv_path = csv_path.with_suffix(".csv")

        # save the DataFrame to CSV
        self.df.to_csv(csv_path, index=False)

    @classmethod
    def from_histories(
        cls,
        loss_history: List[float],
        param_history: List[Parameters],
        epochs: List[int] = None,
        optimizer_history: List[str] = None,
        learning_rate: List[float] = None,
    ) -> "TrainingHistory":
        """Create a TrainingHistory from loss and parameter histories."""
        data = {
            "loss": loss_history,
            **{name: [getattr(p, name) for p in param_history] for name in Parameters._fields},
        }
        df = pd.DataFrame(data)
        if learning_rate is not None:
            df["learning_rate"] = learning_rate
        if optimizer_history is not None:
            df["optimizer"] = optimizer_history
        if epochs is not None:
            df["epoch"] = epochs
        return TrainingHistory(df)

    @classmethod
    def from_csv(cls, csv_path: str) -> "TrainingHistory":
        """Load a TrainingHistory from a CSV file."""
        df = pd.read_csv(csv_path)
        return TrainingHistory(df)

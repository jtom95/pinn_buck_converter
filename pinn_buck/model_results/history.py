from pathlib import Path
from typing import List, NamedTuple, Optional
import pandas as pd
import ast

from ..config import Parameters


class TrainingHistory:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for idx in range(len(self.df)):
            yield self.get_parameters(idx)
            
    def _get_rloads_from_row(self, row: pd.Series) -> List[float]:
        rloads = row.get("Rloads", None)
        if rloads is None:
            raise ValueError("Rloads parameter is missing or not in expected format.")
        if isinstance(rloads, str):
            return ast.literal_eval(rloads)               
        elif isinstance(rloads, list):
            return [float(rload) for rload in rloads]
        elif isinstance(rloads, float):
            return [rloads]
        else:
            Warning("Rloads parameter is not in expected format.")
            return rloads

    def get_parameters(self, idx: int) -> Parameters:
        """Get parameters for a specific iteration."""
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Iteration {idx} out of bounds for history with {len(self.df)} entries.")
        row = self.df.iloc[idx]

        # legacy: check if the df is in the old format
        if row.get("Rloads", None) is None:
            if row.get("Rload1", None) is not None:
                return Parameters.build_from_all_names_iterator(
                    map(
                        lambda x: (x[0], float(x[1])),
                        row.items()
                    )
                )
            else:
                raise ValueError("DataFrame does not contain valid parameter data.")
    
        return Parameters(
            L=float(row.get("L")),
            RL=float(row.get("RL")),
            C=float(row.get("C")),
            RC=float(row.get("RC")),
            Rdson=float(row.get("Rdson")),
            Rloads=self._get_rloads_from_row(row),
            Vin=float(row.get("Vin")),
            VF=float(row.get("VF")),
        )

    def get_latest_parameters(self) -> Parameters:
        return self.get_parameters(len(self.df) - 1)

    def get_best_idx(self, optimizer_type: Optional[str]=None, latest_callback: bool = True) -> int:
        """Get the index of the best loss in the DataFrame."""
        # check if the callback column exists in the DataFrame
        if latest_callback and "callbacks" not in self.df.columns:
            latest_callback = False

        if latest_callback: 
            # keep only the rows with the highest callback count
            df_: pd.DataFrame = self.df[self.df["callbacks"] == self.df["callbacks"].max()]
        else:
            df_ = self.df

        # check if the optimizer type exists in the DataFrame
        if optimizer_type is None:
            return df_["loss"].idxmin()

        if "optimizer" not in self.df.columns:
            raise ValueError("Optimizer type not found in the DataFrame.")
        if optimizer_type not in self.df["optimizer"].unique():
            raise ValueError(f"Optimizer type '{optimizer_type}' not found in the DataFrame. Optimizer types present are: {self.df['optimizer'].unique()}")
        return df_[df_["optimizer"] == optimizer_type]["loss"].idxmin()

    def get_best_parameters(self, optimizer_type: Optional[str]=None, latest_callback: bool = True) -> Parameters:
        """Get the best parameters for a specific optimizer type."""
        best_idx = self.get_best_idx(optimizer_type, latest_callback)
        return self.get_parameters(best_idx)

    def get_best_loss(self, optimizer_type: Optional[str]=None, latest_callback: bool = True) -> float:
        """Get the best loss value for a specific optimizer type."""
        best_idx = self.get_best_idx(optimizer_type, latest_callback)
        return self.df.at[best_idx, "loss"]

    def get_best_epoch(self, optimizer_type: Optional[str]=None, latest_callback: bool = True) -> int:
        """Get the epoch corresponding to the best loss for a specific optimizer type."""
        best_idx = self.get_best_idx(optimizer_type, latest_callback )
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

    def df_from_iterator(self) -> pd.DataFrame:
        """Create a DataFrame from the TrainingHistory iterator."""
        data = {pname: [] for pname in Parameters.get_all_names()}
        for param in self:
            for pname, value in param.iterator():
                data[pname].append(value)

        return pd.DataFrame(data)

    @classmethod
    def from_histories(
        cls,
        loss_history: List[float],
        param_history: List[Parameters],
        epochs: List[int] = None,
        optimizer_history: List[str] = None,
        learning_rate: List[float] = None,
        callbacks: List[int] = None
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
        if callbacks is not None:
            df["callbacks"] = callbacks
        return TrainingHistory(df)

    @classmethod
    def from_csv(cls, csv_path: str) -> "TrainingHistory":
        """Load a TrainingHistory from a CSV file."""
        df = pd.read_csv(csv_path)
        return TrainingHistory(df)

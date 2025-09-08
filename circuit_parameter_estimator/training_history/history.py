from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union, Final
from pathlib import Path
import torch
import pandas as pd
import ast
import json


from ..parameters.parameter_class import Parameters

def _to_scalar(x: Any) -> Union[float, int, None]:
    """Convert scalars and 0-D tensors to Python scalars. Lists/tuples -> list of scalars. None passes through."""
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return float(x.item())
        # non-scalar tensor: store as list for CSV/debugging (you likely won't need this in history)
        return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_scalar(v) for v in x]
    # last resort: try float conversion
    try:
        return float(x)
    except Exception:
        return x  # e.g., strings like optimizer name, keep as-is


class TrainingHistory:
    """
    Generic training history that works with the new Parameters class.

    - Internally stores parameters in FLAT form: ('L', 1e-6), ('Rloads1', 10.0), ...
    - Meta columns (loss/epoch/optimizer/learning_rate/callbacks) are kept separate.
    """
    # ----- small helpers -----
    _META_COLS_DEFAULT = {"loss", "epoch", "optimizer", "learning_rate", "callbacks"}
    DEFAULT_FILE_PATTERN: Final = "*.trhist"

    def __init__(self, df: pd.DataFrame, meta_cols: Optional[Iterable[str]] = None):
        self.df = df
        self._meta_cols = set(meta_cols) if meta_cols is not None else set(self._META_COLS_DEFAULT)

    # -------------- iteration / sizing --------------
    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self):
        for idx in range(len(self.df)):
            yield self.get_parameters(idx)

    # -------------- columns helpers --------------
    @property
    def meta_cols(self) -> List[str]:
        return sorted([c for c in self._meta_cols if c in self.df.columns])

    @property
    def param_cols(self) -> List[str]:
        """All non-meta columns are treated as parameter columns (already flattened)."""
        return [c for c in self.df.columns if c not in self._meta_cols]

    # -------------- accessors --------------
    def get_parameters(self, idx: int) -> "Parameters":
        """Rebuild Parameters for a specific row from flattened parameter columns."""
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Row {idx} out of bounds for history of length {len(self.df)}.")
        row = self.df.iloc[idx]
        flat_items: List[Tuple[str, float]] = []
        for name in self.param_cols:
            val = row[name]
            # If it came back as string (CSV roundtrip), try to parse lists; but we expect flattened scalars here.
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                except Exception:
                    parsed = val
                val = parsed
            # Keep only scalars here; lists shouldn't appear because we store flattened names
            if isinstance(val, list):
                # Rare edge case: if someone saved a list in a param column; flattening would have prevented this.
                raise ValueError(f"Parameter column '{name}' contains a list; expected a scalar.")
            if pd.isna(val):
                # If NaN slipped in, raise to avoid silent bugs
                raise ValueError(f"Parameter column '{name}' is NaN at row {idx}.")
            flat_items.append((name, float(val)))
        return Parameters.build_from_flat(flat_items)

    def get_latest_parameters(self) -> "Parameters":
        return self.get_parameters(len(self.df) - 1)

    # -------------- "best" utilities --------------
    def get_best_idx(
        self, optimizer_type: Optional[str] = None, latest_callback: bool = True
    ) -> int:
        if "loss" not in self.df.columns:
            raise ValueError("Column 'loss' is required to select best index.")

        df_ = self.df
        if latest_callback and "callbacks" in self.df.columns:
            max_cb = self.df["callbacks"].max()
            df_ = self.df[self.df["callbacks"] == max_cb]

        if optimizer_type is not None:
            if "optimizer" not in self.df.columns:
                raise ValueError("Column 'optimizer' not found.")
            if optimizer_type not in self.df["optimizer"].unique():
                raise ValueError(
                    f"Optimizer '{optimizer_type}' not found. Present: {self.df['optimizer'].unique()}."
                )
            df_ = df_[df_["optimizer"] == optimizer_type]

        return int(df_["loss"].idxmin())

    def get_best_parameters(
        self, optimizer_type: Optional[str] = None, latest_callback: bool = True
    ) -> "Parameters":
        return self.get_parameters(self.get_best_idx(optimizer_type, latest_callback))

    def get_best_loss(
        self, optimizer_type: Optional[str] = None, latest_callback: bool = True
    ) -> float:
        return float(self.df.at[self.get_best_idx(optimizer_type, latest_callback), "loss"])

    def get_best_epoch(
        self, optimizer_type: Optional[str] = None, latest_callback: bool = True
    ) -> int:
        idx = self.get_best_idx(optimizer_type, latest_callback)
        if "epoch" not in self.df.columns:
            raise ValueError("Column 'epoch' not found.")
        return int(self.df.at[idx, "epoch"])

    # -------------- mutation / IO --------------
    def drop_columns(self, columns: List[str]) -> "TrainingHistory":
        if isinstance(columns, str):
            columns = [columns]
        self.df.drop(columns=columns, inplace=True, errors="ignore")
        # keep meta set consistent (only matters if you drop meta columns)
        self._meta_cols = {c for c in self._meta_cols if c in self.df.columns}
        return self


    def save(self, path: Path) -> None:
        path = Path(path)
        suffix = self.DEFAULT_FILE_PATTERN[1:] # remove the leading "*"
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.mkdir(parents=True, exist_ok=True)

        # 1) write CSV
        (path / "data.csv").write_text(self.df.to_csv(index=False))

        # 2) write meta.json
        meta = {
            "version": 1,
            "meta_cols": [c for c in self._meta_cols if c in self.df.columns],
            "param_cols": [c for c in self.df.columns if c not in self._meta_cols],
            "dtypes": {c: str(self.df[c].dtype) for c in self.df.columns},
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))


    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":
        path = Path(path)
        suffix = cls.DEFAULT_FILE_PATTERN[1:] # remove the leading "*"
        if path.suffix != suffix:
            raise ValueError(f"Expected a {suffix} folder, got {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")

        # 1) load CSV
        data_path = path / "data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data.csv in {path}")
        df = pd.read_csv(data_path)

        # 2) load meta
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta_cols = set(meta.get("meta_cols", []))
            # optionally validate dtypes
            dtypes = meta.get("dtypes", {})
            for col, dtype_str in dtypes.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype_str)
                    except Exception:
                        # fallback silently if cast fails
                        pass
        else:
            meta_cols = set(cls._META_COLS_DEFAULT)

        return cls(df, meta_cols=meta_cols)
        
    # -------------- construction from histories --------------
    @classmethod
    def from_histories(
        cls,
        loss_history: List[float],
        param_history: List["Parameters"],
        *,
        epochs: Optional[List[int]] = None,
        optimizer_history: Optional[List[str]] = None,
        learning_rate: Optional[List[float]] = None,
        callbacks: Optional[List[int]] = None,
    ) -> "TrainingHistory":
        """
        Build a flat DataFrame out of a list of Parameters snapshots.
        Each param row is built by iterating p.iterator() and casting values to Python scalars.
        """
        n = len(loss_history)
        if len(param_history) != n:
            raise ValueError("param_history and loss_history must have the same length.")
        def _check_len(name: str, seq: List[Any]):
            if len(seq) != n:
                raise ValueError(f"{name} length {len(seq)} != loss_history length {n}")
        
        _check_len("epochs", epochs)
        _check_len("optimizer_history", optimizer_history)
        _check_len("learning_rate", learning_rate)
        _check_len("callbacks", callbacks)

        # Collect all parameter names seen across history (union), in order of first appearance.
        all_param_names: List[str] = []
        seen = set()
        flat_rows: List[Dict[str, Union[float, int, str]]] = []

        for p in param_history:
            row: Dict[str, Union[float, int, str]] = {}
            for name, val in p.iterator():
                if name not in seen:
                    seen.add(name)
                    all_param_names.append(name)
                sval = _to_scalar(val)
                if isinstance(sval, list):
                    # Should not happen (iterator flattens lists), but guard anyway.
                    raise ValueError(f"Iterator yielded list for '{name}'. Expected scalar after flattening.")
                row[name] = sval
            flat_rows.append(row)

        # Build dense table with possible missing names filled by NaN; preserve column order.
        df_params = pd.DataFrame(flat_rows, columns=all_param_names)

        # Meta columns (record exactly which ones we add)
        meta_cols = {"loss"}
        data = {"loss": loss_history}
        if epochs is not None:
            data["epoch"] = epochs
            meta_cols.add("epoch")
        if optimizer_history is not None:
            data["optimizer"] = optimizer_history
            meta_cols.add("optimizer")
        if learning_rate is not None:
            data["learning_rate"] = learning_rate
            meta_cols.add("learning_rate")
        if callbacks is not None:
            data["callbacks"] = callbacks
            meta_cols.add("callbacks")

        df_meta = pd.DataFrame(data)

        # Align by index to avoid accidental misalignment if anything is off.
        df_meta.index = range(n)
        df_params.index = range(n)

        df = pd.concat([df_meta, df_params], axis=1)

        return cls(df, meta_cols=meta_cols)


    # -------------- optional: convenience export of just the params --------------
    def params_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with only parameter columns."""
        return self.df[self.param_cols].copy()

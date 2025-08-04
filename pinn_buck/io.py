from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt

@dataclass
class TransientData:
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    i: np.ndarray = field(default_factory=lambda: np.array([]))
    dt: np.ndarray = field(default_factory=lambda: np.array([]))
    D: np.ndarray = field(default_factory=lambda: np.array([]))

    def plot_transient(
        self, slice_index: slice, ax: List[plt.Axes] = None, label: str = None, 
        markers=None, linestyle="-",
        figsize=(8, 3), include_D: bool = False, legend: bool = True, 
        color=None, D_color="orange", D_weight=0.5,
        legend_loc: str = None, ignore_dt: bool = False, markersize=None, grid: bool = False
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
            


        time = self.time[slice_index] if slice_index is not None else self.time
        v = self.v[slice_index] if slice_index is not None else self.time
        i = self.i[slice_index] if slice_index is not None else self.time
        dt = self.dt[slice_index] if slice_index is not None else self.time
        if include_D:
            D = self.D[slice_index] if slice_index is not None else self.time
                
                
        ax[0].plot(time, v, label=label, color=color, marker=markers, linestyle=linestyle, markersize=markersize)
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.0f}"))

        ax[0].set_xlabel("Time (us)")
        ax[0].set_ylabel("Voltage (V)")
        ax[0].set_title("Voltage Transient")

        ax[1].plot(time, i, label=label, color=color, marker=markers, linestyle=linestyle, markersize=markersize)
        ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.0f}"))
        ax[1].set_xlabel("Time (us)")
        ax[1].set_ylabel("Current (A)")
        ax[1].set_title("Current Transient")

        if not ignore_dt:
            ax[2].plot(time, dt, label=label, color=color, marker=markers, linestyle=linestyle, markersize=markersize)
            ax[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.0f}"))
            ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.2f}"))
            ax[2].set_xlabel("Time (us)")
            ax[2].set_ylabel("Time Step (s)")
            ax[2].set_title("Time Step Transient")

        if include_D:
            # add a square wave plot for D
            D_square_wave_x, D_square_wave_y = self.to_square_wave(D)
            # transform the D_square_wave_x to match the time array
            D_square_wave_x = (D_square_wave_x - D_square_wave_x[0]) / (
                D_square_wave_x[-1] - D_square_wave_x[0]
            ) * (time[-1] - time[0]) + time[0]

            # the square wave should be plotted with respect to a second y-axis on the right
            for axx in ax:
                axx.twinx().step(
                    D_square_wave_x,
                    D_square_wave_y,
                    label="D",
                    color=D_color,
                    where="post",
                    linewidth=D_weight,
                    linestyle="--",
                )
        if legend:
            for axx in ax:
                if legend_loc is None:
                    axx.legend()
                else:
                    axx.legend(loc=legend_loc)
        if grid:
            for axx in ax:
                axx.grid(True)

    @staticmethod
    def to_square_wave(arr):
        # Repeat each value, except the last, to create a step effect
        arr = np.asarray(arr)
        x = np.repeat(np.arange(len(arr)), 2)[1:]
        y = np.repeat(arr, 2)[:-1]
        return x, y

    @classmethod
    def from_h5(cls, h5_file: str, dataset_name: str, transient_number: int) -> "TransientData":
        with h5py.File(h5_file, "r") as f:
            data = f[dataset_name][f"subtransient_{transient_number}"]
            time = data["t"][:]
            v = data["v"][:]
            i = data["i"][:]
            dt = data["dt"][:]
            D = data["Dswitch"][:]
        return cls(time, v, i, dt, D)


class Measurement:
    def __init__(self, transients: List[TransientData]):
        self.transients = transients
        self.transient_lengths = [len(tr.i) for tr in transients]

        # usually we have 3 transients
    @property
    def tr1(self) -> TransientData:
        if len(self.transients) > 0:
            return self.transients[0]
        return None
    @tr1.setter
    def tr1(self, transient: TransientData):
        if len(self.transients) == 0:
            self.transients.append(transient)
        else:
            self.transients[0] = transient
    @property
    def tr2(self) -> TransientData:
        if len(self.transients) > 1:
            return self.transients[1]
        return None
    @tr2.setter
    def tr2(self, transient: TransientData):
        if len(self.transients) == 0:
            raise ValueError("No transients loaded. Cannot set tr2: First tr1 must be set.")
        if len(self.transients) == 1:
            self.transients.append(transient)
        else:
            self.transients[1] = transient
    @property
    def tr3(self) -> TransientData:
        if len(self.transients) > 2:
            return self.transients[2]
        return None
    @tr3.setter
    def tr3(self, transient: TransientData):
        if len(self.transients) == 0:
            raise ValueError("No transients loaded. Cannot set tr3: First tr1 and tr2 must be set.")    
        if len(self.transients) == 1:
            raise ValueError("No tr2 loaded. Cannot set tr3: First tr2 must be set.")
        if len(self.transients) == 2:
            self.transients.append(transient)
        else:
            self.transients[2] = transient

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        X_parts, y_parts = [], []
        for tr in self.transients:
            i, v, d, dt = (
                tr.i,
                tr.v,
                tr.D,
                tr.dt,
            )
            x = np.hstack([i[:-1, None], v[:-1, None], d[:-1, None], dt[:-1, None]])
            y = np.hstack([i[1:, None], v[1:, None]])
            X_parts.append(x)
            y_parts.append(y)
        X = np.vstack(X_parts).astype(np.float32)
        y = np.vstack(y_parts).astype(np.float32)
        return X, y
    
    @property
    def transient_idx(self) -> np.ndarray:
        """Return the transient index for each row in the data."""
        idx_parts = []
        for ii, tr in enumerate(self.transients):
            n = len(tr.i) - 1  # exclude the last point since it has no next value
            idx_parts.append(np.full((n, 1), ii, dtype=np.int64))
        return np.vstack(idx_parts) if idx_parts else np.empty((0, 1), dtype=np.int64)

    def plot_data(
        self,
        slice_index: slice = None,
        ax: List[plt.Axes] = None,
        label: str = None,
        figsize=(8, 7),
        include_D: bool = False,
        markers=None,
        markersize=None,
        linestyle=None,
        sharex=True,
        color=None,
        legend=True,
        D_weight=0.5,
        legend_loc: str = None,
        ignore_dt: bool = False,
        grid: bool = False
    ) -> plt.Axes:
        """Plot the loaded measurements."""
        # check the number of transients loaded
        n_transients = len(self.transients)
        if n_transients == 0:
            raise ValueError("No measurements loaded. Call load() first.")

        n_plots = 2 if ignore_dt else 3

        if ax is None:
            fig, ax = plt.subplots(
                n_transients,
                n_plots + 1,
                figsize=figsize,
                constrained_layout=True,
                gridspec_kw={"width_ratios": [0.05] + [1] * n_plots},
                sharex=sharex,
            )
        else:
            fig = ax[0, 0].figure

        for idx, axx in enumerate(ax[:, 0]):
            axx.axis("off")
            title = f"Transient {idx + 1}"
            # add the title as text in the first column
            axx.text(
                0.5,
                0.5,
                title,
                fontsize=12,
                ha="center",
                va="center",
                transform=axx.transAxes,
                rotation=90,
            )

        for idx, tr in enumerate(self.transients):
            tr.plot_transient(
                ax=ax[idx, 1:],
                label=label,
                markers=markers,
                linestyle=linestyle,
                figsize=figsize,
                include_D=include_D,
                color=color,
                slice_index=slice_index,
                D_weight=D_weight,
                legend=legend,
                legend_loc=legend_loc,
                ignore_dt=ignore_dt,
                markersize=markersize,
                grid=grid,
            )

        # remove the titles from the rows after the first one
        for axx in ax[1:, :].flatten():
            axx.set_title("")
        for axx in ax[:-1, :].flatten():
            if sharex:
                axx.set_xlabel("")
        return ax
    
    def save_to_numpyzip(self, filepath: Path):
        """Save measurements to a numpyz file."""
        if not filepath.parent.exists():
            raise FileNotFoundError(f"Directory {filepath.parent} does not exist.")
        data = {f"tr{ii + 1}": (tr.time, tr.v, tr.i, tr.dt, tr.D) for ii, tr in enumerate(self.transients)}
        np.savez_compressed(filepath, **data)
        
    @classmethod
    def load_from_numpyzip(cls, filepath: Path) -> "Measurement":
        """Load measurements from a numpyz file."""
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        with np.load(filepath, allow_pickle=True) as data:
            transients = []
            for key in data.files:
                time, v, i, dt, D = data[key]
                transients.append(TransientData(time=time, v=v, i=i, dt=dt, D=D))
        return cls(transients)
        

class LoaderH5:
    def __init__(self, db_dir: Path, h5filename: str):
        if not db_dir.exists():
            raise FileNotFoundError(f"Database directory {db_dir} does not exist.")
        if not (db_dir / h5filename).exists():
            raise FileNotFoundError(f"HDF5 file {h5filename} does not exist in {db_dir}.")

        self.db_dir = db_dir
        self.h5filename = h5filename
        self.measurements_dict: Dict[str, TransientData] = {}
    
    def load(self, measurement_name: str) -> TransientData:
        """Load measurements from an HDF5 file."""
        self.measurements_dict: Dict[str, TransientData] = self._load_measurements(
            measurement_name, filepath=self.db_dir / self.h5filename
        )
        for ii, k in enumerate(self.measurements_dict.keys()):
            # check if attribute already exists, if not, create it            
            setattr(self, f"tr{ii + 1}", self.measurements_dict[k])
            
    @property
    def M(self) -> Measurement:
        if len(self.measurements_dict) == 0:
            raise ValueError("No measurements loaded. Call load() first.")
        return Measurement(list(self.measurements_dict.values()))

    @staticmethod
    def _load_measurements(measurement_name: str, filepath: Path) -> Dict[str, TransientData]:
        """Load measurements from an HDF5 file."""

        def _generate_TransientData(key: str, g: h5py.Group) -> TransientData:
            i = g[key]["i"][:]
            v = g[key]["v"][:]
            dt = g[key]["dt"][:]
            Dswitch = g[key]["Dswitch"][:]
            # check if t is present, if not, generate it
            if "t" in g[key]:
                t = g[key]["t"][:]
            else:
                # generate time based on dt
                t = np.cumsum(dt)
                t = np.insert(t, 0, 0)

            return TransientData(
                time=t,
                v=v,
                i=i,
                dt=dt,
                D=Dswitch,
            )

        with h5py.File(filepath, "r") as f:
            g = f[measurement_name]
            S = {k: _generate_TransientData(k, g) for k in g.keys()}
        return S

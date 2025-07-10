from .io import TransientData, Measurement
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path
from .io_model import TrainingRun


def add_noise_to_TransientData(
    tr: TransientData,
    noise_level: float,
    normalize_to_LSB: bool = True,
    vmax: float = None,
    vmin: float = None,
    imax: float = None,
    imin: float = None,
) -> TransientData:
    """
    Add noise to the current and voltage data of a TransientData object.
    """
    i = tr.i.copy()
    v = tr.v.copy()

    if normalize_to_LSB:
        vmax = vmax if vmax is not None else tr.v.max()
        vmin = vmin if vmin is not None else tr.v.min()
        imax = imax if imax is not None else tr.i.max()
        imin = imin if imin is not None else tr.i.min()

        LSB_i = (imax - imin) / (2**12 - 1)  # assuming 12-bit ADC
        LSB_v = (vmax - vmin) / (2**12 - 1)  # assuming 12-bit ADC

        noise_level_i = noise_level * LSB_i  # normalize noise level to LSB
        noise_level_v = noise_level * LSB_v  # normalize noise level to LSB
    else:
        noise_level_v = noise_level  # keep noise level in the same unit as the data
        noise_level_i = noise_level  # keep noise level in the same unit as the data

    i += np.random.normal(0, noise_level_i, size=i.shape)
    v += np.random.normal(0, noise_level_v, size=v.shape)
    return TransientData(time=tr.time, i=i, v=v, dt=tr.dt, D=tr.D)


def add_noise_to_Measurement(
    M: Measurement,
    noise_level: float,
    V_FS: int = 10,
    I_FS: int = 30,
    scale_transients_together: bool = False,
    scale_transients_independently: bool = False,
) -> Measurement:
    """
    Add noise to the current and voltage data of a Measurement object.

    If scale_transients_together is True, a common min/max is used to compute LSB across all transients.
    If scale_transients_independently is True, each transient uses its own local min/max for LSB computation.
    If both are False (default), fixed full-scale ranges (±V_FS/2, ±I_FS/2) are used.
    """
    noisy_tr = []

    if scale_transients_together:
        v = np.concatenate([tr.v for tr in M.transients])
        i = np.concatenate([tr.i for tr in M.transients])
        vmax = v.max()
        vmin = v.min()
        imax = i.max()
        imin = i.min()
    elif scale_transients_independently:
        vmax = None
        vmin = None
        imax = None
        imin = None
    else: 
        vmax = V_FS/2
        vmin = -V_FS/2
        imax = I_FS/2
        imin = -I_FS/2

    for tr in M.transients:
        noisy_tr.append(
            add_noise_to_TransientData(
                tr,
                noise_level=noise_level,
                vmax=vmax,
                vmin=vmin,
                imax=imax,
                imin=imin,
            )
        )
    return Measurement(noisy_tr)


def get_repeated_noisy_runs(run_dir: Path):
    """
    Function to get repeated noisy runs from a specified directory.
    This function is not used in the current script but can be useful for future modifications.
    """
    csv_files = list(run_dir.glob("*.csv"))
    runs = {}
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}")
        noise_level = float(csv_file.stem.split("_")[-2])  # Extract noise level from filename
        if noise_level not in runs:
            tr = TrainingRun.from_csv(csv_file)
            runs[noise_level] = [tr.drop_columns(["learning_rate"])]

        else:
            tr = TrainingRun.from_csv(csv_file)
            runs[noise_level].append(tr.drop_columns(["learning_rate"]))
    return runs


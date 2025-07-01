from .io import TransientData, Measurement
import numpy as np
from typing import List
import matplotlib.pyplot as plt

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


def inspect_repeated_lossy_data(
    lossy_meas: List[Measurement],
    label: str,
    reference: Measurement,
    slice_index: slice,
    ax=None,
    figsize=(12, 10),
    markersize=2,
    color="blue",
    plot_ideal: bool = True,
) -> List[plt.Axes]:
    """
    Inspect the repeated lossy data by plotting the measurements.
    """
    if plot_ideal:
        ax = reference.plot_data(
            label="ideal",
            sharex=True,
            ax=ax,
            slice_index=slice_index,
            legend=True,
            figsize=figsize,
            color="black",
            ignore_dt=True,
        )

    lossy_label = label
    for idx, meas in enumerate(lossy_meas):
        meas.plot_data(
            label=lossy_label,
            ax=ax,
            slice_index=slice_index,
            legend=True,
            markers=".",
            linestyle=" ",
            ignore_dt=True,
            markersize=markersize,
            color=color,
        )
        lossy_label = None
    return ax

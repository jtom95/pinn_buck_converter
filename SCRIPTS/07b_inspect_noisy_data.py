from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import h5py
import numpy as np

# ---------------------------------------------------------------------
# Parameters container & scaling helpers
# ---------------------------------------------------------------------
# add the root directory of the project to the path
sys.path.append(str(Path(__file__).parent.parent))

from pinn_buck.config import NOMINAL as NOMINAL_PARAMS, INITIAL_GUESS as INITIAL_GUESS_PARAMS
from pinn_buck.config import _SCALE

from pinn_buck.noise import add_noise_to_Measurement, inspect_repeated_lossy_data

from pinn_buck.io import LoaderH5, Measurement


# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"

io = LoaderH5(db_dir, h5filename)
io.load("ideal")

ideal_meas = io.M

out_dir = (
    Path(__file__).parent.parent / "RESULTS" / "Adam_Opt" / "noisy_runs" / "DATA"
)
out_dir.mkdir(parents=True, exist_ok=True)


noisy_measurements: Dict[int, List[Measurement]] = {}
for file in out_dir.glob("*.npz"):
    # noise_level
    noise_level = float(
        file.name.split("_")[1].replace("LSB", "")
    )  # Extract noise level from filename
    if noise_level not in noisy_measurements:
        noisy_measurements[noise_level] = [Measurement.load_from_numpyzip(file)]
    else:
        noisy_measurements[noise_level].append(Measurement.load_from_numpyzip(file))

# reorder the measurements by noise level
noise_levels = sorted(noisy_measurements.keys())
noisy_measurements = {noise: noisy_measurements[noise] for noise in noise_levels}



# choose a palette of colors for the noisy measurements
import seaborn as sns
colors = sns.color_palette("husl", len(noisy_measurements))

ax = None
plot_ideal = True
for i, noise_level in enumerate(noise_levels[::-1]):
    measurements = noisy_measurements[noise_level]
    # Plot the noisy measurements
    ax = inspect_repeated_lossy_data(
        measurements,
        label=f"{noise_level} LSB",
        reference=ideal_meas,
        slice_index=slice(0, 30),
        figsize=(12, 10),
        markersize=5,
        color=colors[i],
        ax = ax,
        plot_ideal=plot_ideal
    )
    plot_ideal = False  # Only plot the ideal measurement once
    
    
plt.show(block = True)
print("Done inspecting noisy data.")


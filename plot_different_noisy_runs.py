import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple
from pathlib import Path
import sys

# sys.path.append(str(Path(__file__).parent.parent))

from pinn_buck.config import Parameters, TrainingRun, NOMINAL
from pinn_buck.plot_utils import plot_tracked_parameters, plot_final_percentage_error, plot_final_percentage_error_multi

run_dir = Path(__file__).parent / "RESULTS" / "removed_nn" / "noisy_runs"

# loop through all CSV files in the directory
csv_files = list(run_dir.glob("*.csv"))
runs = {}
for csv_file in csv_files:
    print(f"Processing {csv_file.name}")
    noise_level = float(csv_file.stem.split("_")[-1])  # Extract noise level from filename
    tr = TrainingRun.from_csv(csv_file)
    runs[f"{noise_level}*LSB"] = tr.drop_columns(["learning_rate"])  # drop learning rate column

run_ideal = TrainingRun.from_csv(run_dir.parent / "rk4.csv").drop_columns(["learning_rate"])  

fig, ax = plot_tracked_parameters(
    df=run_ideal,
    target=NOMINAL,
    label="0",
    color="black",
    figsize=(18, 10),
)

ordered_runs = sorted(runs.items(), key=lambda x: float(x[0].split(" ")[-1].replace("*LSB", "")))

for label, tr in ordered_runs:
    plot_tracked_parameters(
        df=tr,
        target=None,
        label=label,
        ax=ax,
        color=None
    )


plot_final_percentage_error_multi(
    runs={"0": run_ideal, **runs},
    target=NOMINAL,
    figsize=(14, 5)
)


# tr = TrainingRun.from_csv("RESULTS/removed_nn/only_current.csv")

# # # discard the first 1000 iterations for better visualization
# # df = df.iloc[3:].reset_index(drop=True)

# plot_tracked_parameters(tr, target=NOMINAL)
# plot_final_percentage_error(tr, target=NOMINAL)
plt.show()

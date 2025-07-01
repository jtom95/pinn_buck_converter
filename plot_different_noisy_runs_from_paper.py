import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple
from pathlib import Path
import sys

# sys.path.append(str(Path(__file__).parent.parent))

from pinn_buck.config import Parameters, TrainingRun, NOMINAL
from pinn_buck.plot_utils import plot_tracked_parameters, plot_final_percentage_error, plot_final_percentage_error_multi

run_dir = Path(__file__).parent / "RESULTS" / "removed_nn" / "noisy_runs_from_paper_composed_loss"

# loop through all CSV files in the directory
csv_files = list(run_dir.glob("*.csv"))
runs = {}
for csv_file in csv_files:
    print(f"Processing {csv_file.name}")
    # noise_level = float(csv_file.stem.split("_")[-1])  # Extract noise level from filename
    # tr = TrainingRun.from_csv(csv_file)
    # runs[f"{noise_level}*LSB"] = tr.drop_columns(["learning_rate"])  # drop learning rate column

    label = csv_file.stem.removeprefix("noisy_run_")  # Extract label from filename
    tr = TrainingRun.from_csv(csv_file)
    runs[label] = tr.drop_columns(["learning_rate"])  # drop learning rate column

GROUP_NUMBER_DICT = {
    0: "ideal",
    1: "ADC_error",
    2: "Sync Error",
    3: "5 noise",
    4: "10 noise",
    5: "ADC-Sync-5noise",
    6: "ADC-Sync-10noise",
}


for ii in (0, 1, 3, 4):
    tr = runs[GROUP_NUMBER_DICT[ii]]
    label = GROUP_NUMBER_DICT[ii]
    if ii ==0: 
       fig, ax = plot_tracked_parameters(
            df=tr,
            target=NOMINAL,
            label=label,
            color="black",
            figsize=(18, 10),
        )
       continue
    
    plot_tracked_parameters(
        df=tr,
        target=None,
        label=label,
        ax=ax,
        color=None
    )

runs_ordered = {GROUP_NUMBER_DICT[ii]: runs[GROUP_NUMBER_DICT[ii]] for ii in (0, 1, 3, 4)}

plot_final_percentage_error_multi(
    runs=runs_ordered,
    target=NOMINAL,
    figsize=(14, 5),
    select_lowest_loss=False
)


# tr = TrainingRun.from_csv("RESULTS/removed_nn/only_current.csv")

# # # discard the first 1000 iterations for better visualization
# # df = df.iloc[3:].reset_index(drop=True)

# plot_tracked_parameters(tr, target=NOMINAL)
# plot_final_percentage_error(tr, target=NOMINAL)
plt.show()

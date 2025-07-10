import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import NamedTuple, List, Iterable, Dict
from pathlib import Path
import sys

# sys.path.append(str(Path(__file__).parent.parent))

from pinn_buck.config import Parameters, TRUE
from pinn_buck.io_model import TrainingRun
from pinn_buck.noise import get_repeated_noisy_runs
from pinn_buck.plot_utils import plot_tracked_parameters, plot_final_percentage_error, plot_final_percentage_error_multi
from pinn_buck.plot_utils_noisy import plot_final_percentage_error_multi_boxplot, plot_repeated_tracked_parameters

run_dir = Path(__file__).parent / "RESULTS" / "Adam_Opt" / "noisy_runs" / "weightedloss"


run_dict = get_repeated_noisy_runs(run_dir)
# noise_level = 3  # Example noise level to plot

ax = None
for noise_level, runs in run_dict.items():
    fig, ax = plot_repeated_tracked_parameters(
        runs = run_dict[noise_level],
        target=TRUE,
        label=f"{noise_level}*LSB",
        color=None,
        figsize=(18, 10),
        ax=ax,
    )

# select a bright color palette for the boxplot
palette = sns.color_palette("bright", n_colors=len(run_dict))

plot_final_percentage_error_multi_boxplot(
    runs=run_dict, target=TRUE, figsize=(14, 5), select_lowest_loss=False, palette=palette
)

# ordered_runs = sorted(runs.items(), key=lambda x: float(x[0].split(" ")[-1].replace("*LSB", "")))

# for label, tr in ordered_runs:
#     plot_tracked_parameters(
#         df=tr,
#         target=None,
#         label=label,
#         ax=ax,
#         color=None
#     )


# plot_final_percentage_error_multi(
#     runs={"0": run_ideal, **runs},
#     target=TRUE,
#     figsize=(14, 5)
# )


# tr = TrainingRun.from_csv("RESULTS/removed_nn/only_current.csv")

# # # discard the first 1000 iterations for better visualization
# # df = df.iloc[3:].reset_index(drop=True)

# plot_tracked_parameters(tr, target=TRUE)
# plot_final_percentage_error(tr, target=TRUE)
plt.show()
print("Stop")

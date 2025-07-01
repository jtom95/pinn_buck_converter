import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple
from pathlib import Path
import sys

# sys.path.append(str(Path(__file__).parent.parent))

from pinn_buck.config import Parameters, TrainingRun, NOMINAL
from pinn_buck.plot_utils import plot_tracked_parameters, plot_final_percentage_error

tr = TrainingRun.from_csv("RESULTS/removed_nn/only_current.csv")

# # discard the first 1000 iterations for better visualization
# df = df.iloc[3:].reset_index(drop=True)

plot_tracked_parameters(tr, target=NOMINAL)
plot_final_percentage_error(tr, target=NOMINAL)
plt.show()


# %%
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import h5py
import numpy as np


# change the working directory to the root of the project
sys.path.append(str(Path.cwd().parent))


from pinn_buck.config import Parameters

# load measurement interface
from pinn_buck.io import Measurement
from pinn_buck.noise import add_noise_to_Measurement

from pinn_buck.parameter_transformation import make_log_param, reverse_log_param
from pinn_buck.model.model_param_estimator import BuckParamEstimator
from pinn_buck.model.trainer import Trainer


from pinn_buck.io import LoaderH5

# %%
import matplotlib
from scipy.stats import lognorm
from pinn_buck.constants import ParameterConstants, MeasurementGroupArchive
from pinn_buck.data_noise_modeling.auxiliary import rel_tolerance_to_sigma

TRUE_PARAMS = ParameterConstants.TRUE
NOMINAL = ParameterConstants.NOMINAL
REL_TOL = ParameterConstants.REL_TOL
PRIOR_SIGMA = rel_tolerance_to_sigma(REL_TOL, number_of_stds_in_relative_tolerance=1) # transforms relative tolerance to the value of the standard deviation

# print the nominal parameters
print("Nominal Parameters:")
print(NOMINAL)

print("Relative Tolerances:")
print(REL_TOL)


# %%
from pinn_buck.model_results.history import TrainingHistory
from pinn_buck.model_results.ploting_comparisons import ResultsComparerTwo
# from pinn_buck.plot_aux import place_shared_legend


out_dir = Path.cwd() / "RESULTS" / "DiagonalBlocksCovariance"


# %%
rc = ResultsComparerTwo.from_dirs({"BlockDiagonal": out_dir})

# Choose the labels you care about (ints map via the default dict; strings work too)
labels = ("ADC_error", "5 noise", "10 noise")

# Final % error comparison (side-by-side)
fig, ax = rc.plot_comparison(labels=labels, target=TRUE_PARAMS, select_lowest_loss=False)

# Optional: tracked parameters for specific curves


fig, axes = rc.plot_tracked(target=TRUE_PARAMS, labels=labels)
plt.show(block=True)
print("done")

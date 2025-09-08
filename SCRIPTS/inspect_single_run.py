import sys
from pathlib import Path
import matplotlib.pyplot as plt

# change the working directory to the root of the project
sys.path.append(str(Path.cwd()))

from circuit_parameter_estimator.data_covariance.auxiliary import rel_tolerance_to_sigma
from circuit_parameter_estimator.laplace_posterior.plotting import LaplacePosteriorPlotter
from circuit_parameter_estimator.examples_archive.buck_converter import ParameterArchive, MeasurementGroupArchive

TRUE_PARAMS = ParameterArchive.TRUE
NOMINAL = ParameterArchive.NOMINAL
REL_TOL = ParameterArchive.REL_TOL
PRIOR_SIGMA = rel_tolerance_to_sigma(
    REL_TOL, number_of_stds_in_relative_tolerance=1
)  # transforms relative tolerance to the value of the standard deviation


# from pinn_buck.plot_aux import place_shared_legend

results_directory = Path.cwd() / "RESULTS" / "LIKELIHOODS"
save_dir = results_directory / "FWD"

lplotter = LaplacePosteriorPlotter.from_dir(
    save_dir,
    group_number_dict=MeasurementGroupArchive.SHUAI_ORIGINAL,
)

lplotter.plot_laplace_posteriors(true_params=TRUE_PARAMS, ncols=3)
lplotter.plot_uncertainty_percent()
lplotter.plot_ci(n_sigma=1.5, ncols=3, true_params=TRUE_PARAMS)
plt.show()
print(lplotter.ci_dataframe(1))

# lfits = LaplacePosteriorPlotter.load_lfits_from_dir(save_dir)

# LaplacePosteriorPlotter.ci_dataframe(lfits).head()
print("done")

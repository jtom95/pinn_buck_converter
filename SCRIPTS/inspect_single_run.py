import sys
from pathlib import Path
import matplotlib.pyplot as plt

# change the working directory to the root of the project
sys.path.append(str(Path.cwd()))

from circuit_parameter_estimator.model_results.history import TrainingHistory
from circuit_parameter_estimator.model_results.ploting_comparisons import ResultsComparerTwo
from circuit_parameter_estimator.constants import ParameterConstants, MeasurementGroupArchive
from circuit_parameter_estimator.data_covariance.auxiliary import rel_tolerance_to_sigma
from circuit_parameter_estimator.laplace_posterior_fitting import LaplacePosterior
from circuit_parameter_estimator.laplace_posterior_plotting import LaplacePosteriorPlotter, LaplaceDictionaryLoader
from circuit_parameter_estimator.laplace_posterior_plotting_comparison import LaplaceResultsComparer


TRUE_PARAMS = ParameterConstants.TRUE
NOMINAL = ParameterConstants.NOMINAL
REL_TOL = ParameterConstants.REL_TOL
PRIOR_SIGMA = rel_tolerance_to_sigma(
    REL_TOL, number_of_stds_in_relative_tolerance=1
)  # transforms relative tolerance to the value of the standard deviation


# from pinn_buck.plot_aux import place_shared_legend

results_directory = Path.cwd() / "RESULTS" / "LIKELIHOODS"
save_dir = results_directory / "FWD"

lplotter = LaplacePosteriorPlotter.from_dir(
    save_dir
)

lplotter.plot_laplace_posteriors(true_params=TRUE_PARAMS, ncols=3)
lplotter.plot_uncertainty_percent()
lplotter.plot_ci(n_sigma=1.5, ncols=3, true_params=TRUE_PARAMS)
plt.show()
print(lplotter.ci_dataframe(1))

# lfits = LaplacePosteriorPlotter.load_lfits_from_dir(save_dir)

# LaplacePosteriorPlotter.ci_dataframe(lfits).head()
print("done")

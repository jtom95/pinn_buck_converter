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
from circuit_parameter_estimator.laplace_posterior_plotting import LaplacePosteriorPlotter
from circuit_parameter_estimator.laplace_posterior_plotting_comparison import LaplaceResultsComparer


TRUE_PARAMS = ParameterConstants.TRUE
NOMINAL = ParameterConstants.NOMINAL
REL_TOL = ParameterConstants.REL_TOL
PRIOR_SIGMA = rel_tolerance_to_sigma(
    REL_TOL, number_of_stds_in_relative_tolerance=1
)  # transforms relative tolerance to the value of the standard deviation


# from pinn_buck.plot_aux import place_shared_legend

directory_dict = {
    "fwd_vif": Path.cwd() / "RESULTS" / "LIKELIHOODS" / "FWD_VIF",
    "fwd": Path.cwd() / "RESULTS" / "LIKELIHOODS" / "FWD",
}

rc = ResultsComparerTwo.from_dirs(
    directory_dict
)

# Choose the labels you care about (ints map via the default dict; strings work too)
labels = ("ADC_error", "5 noise", "10 noise")

# # Final % error comparison (side-by-side)
fig, ax = rc.plot_comparison(
    labels=labels, target=TRUE_PARAMS, select_lowest_loss=True, legend_bottom_inch=0.15
)

# # Optional: tracked parameters for specific curves

fig, axes = rc.plot_tracked(target=TRUE_PARAMS, labels=labels)

laplace_comparer = LaplaceResultsComparer.from_dirs(directory_dict)
laplace_comparer.plot_ci(ncols=4)
laplace_comparer.plot_posteriors_grid(
    skip_labels=("ADC_error",),
    ncols=4, 
    prior_mu=NOMINAL, 
    prior_sigma=PRIOR_SIGMA, 
    true_params=TRUE_PARAMS,
    )
laplace_comparer.plot_param_overlay("Rdson")
laplace_comparer.plot_param_overlay("Rdson", prior_mu=NOMINAL, prior_sigma=PRIOR_SIGMA, true_params=TRUE_PARAMS,)
plt.show()
print("done")
import torch

from pinn_buck.model import trainer
from ..config import Parameters
from .model_param_estimator import BaseBuckEstimator
from .residuals import ResidualFunc
from .residual_time_correlation import ResidualDiagnosticsGaussian
from ..data_noise_modeling.covariance_matrix_config import (
    DataCovariance,
    ResidualCovarianceMatrixFunc,
)
from .loss_function_configs import LikelihoodLossFunction, PriorLossFunction

from .loss_function_archive import (
    log_normal_prior,
    loss_whitened,
)
from ..data_noise_modeling.covariance_matrix_function_archive import (
    chol,
    generate_residual_covariance_matrix,
)
from ..data_noise_modeling.jacobian_estimation import JacobianEstimatorBase, FwdBckJacobianEstimator

def calculate_covariance_matrix(
    model: BaseBuckEstimator,
    X: torch.Tensor,
    data_covariance: torch.Tensor,
    jacobian_estimator: JacobianEstimatorBase,
    residual_covariance_func: ResidualCovarianceMatrixFunc,
    number_of_samples_for_jacobian_estimation: int = 500,
    damp_residual_covariance_matrix: float = 1e-10,
):

    jac = jacobian_estimator.estimate_Jacobian(
        X, model, number_of_samples=number_of_samples_for_jacobian_estimation, dtype=torch.float64
    )[
        ..., :2, :2
    ]  # keep a size of (T, 2, 2)

    # work also for fwd, bck models
    
    if jac.dim == 2: 
        jac = jac[None, None, ...]
    elif jac.dim == 3:
        jac = jac[None,  ...]
    else:
        raise ValueError(f"Jacobian must have 2 or 3 dimensions, got {jac.dim()} dimensions.")

    cov_matrices = []

    for jac_i in jac:    
        covariance_matrix = generate_residual_covariance_matrix(
            data_covariance=data_covariance,
            residual_covariance_func=residual_covariance_func,
            jac=jac_i,
            damp=damp_residual_covariance_matrix,
            dtype=torch.float64,
        )
        cov_matrices.append(covariance_matrix)
    return torch.stack(cov_matrices, dim=0) if len(cov_matrices) > 1 else cov_matrices[0]


def calculate_inflation_factor(
    model: BaseBuckEstimator, 
    X: torch.Tensor, 
    Lr: torch.Tensor, 
    residual_func: ResidualFunc,
    max_lags_vif: int = 500,
    use_std_renorm: bool = True
) -> torch.Tensor:

    with torch.no_grad():
        preds = model(X)
        targets = model.targets(X)
        # Extract residuals: (N, T, 2)
        residuals = residual_func(preds, targets)

    diag = ResidualDiagnosticsGaussian(residuals)
    vif = diag.quadloss_vif_from_residuals(Lr, max_lag=max_lags_vif, use_std_renorm=use_std_renorm)
    return vif


###################################################################################

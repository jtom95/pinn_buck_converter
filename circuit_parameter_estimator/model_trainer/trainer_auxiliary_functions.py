import torch

from circuit_parameter_estimator.model import trainer
from ..parameters.parameter_class import Parameters
from ..model.model_base import BaseBuckEstimator
from ..residuals.residuals import ResidualFunc
from ..residuals.residual_time_correlation import ResidualDiagnosticsGaussian
from ..data_covariance.covariance_matrix_config import (
    DataCovariance,
    ResidualCovarianceMatrixFunc,
)

from ..data_covariance.covariance_matrix_function_archive import (
    chol,
    generate_residual_covariance_matrix,
)
from ..data_covariance.jacobian_estimation import JacobianEstimatorBase, FwdBckJacobianEstimator

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
    
    covariance_matrix = generate_residual_covariance_matrix(
            data_covariance=data_covariance,
            residual_covariance_func=residual_covariance_func,
            jac=jac,
            damp=damp_residual_covariance_matrix,
            dtype=torch.float64,
        )
    return covariance_matrix

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



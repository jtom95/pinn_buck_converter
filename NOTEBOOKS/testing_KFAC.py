# %%
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import h5py
import numpy as np

from kfac.preconditioner import KFACPreconditioner


# change the working directory to the root of the project
sys.path.append(str(Path.cwd().parent))


from pinn_buck.config import Parameters
from pinn_buck.config import TRUE as TRUE_PARAMS, INITIAL_GUESS as INITIAL_GUESS_PARAMS
from pinn_buck.config import _SCALE

# load measurement interface
from pinn_buck.io import Measurement
from pinn_buck.noise import add_noise_to_Measurement

from pinn_buck.parameter_transformation import make_log_param, reverse_log_param
from pinn_buck.model.model_param_estimator import BuckParamEstimator
from pinn_buck.model.losses import l2_loss
from pinn_buck.io_model import TrainingRun

from pinn_buck.io import LoaderH5


# %%
import matplotlib
from scipy.stats import lognorm
from pinn_buck.config import Parameters


# Nominals and linear-space relative tolerances
NOMINAL = Parameters(
    L=6.8e-4,
    RL=0.4,
    C=1.5e-4,
    RC=0.25,
    Rdson=0.25,
    Rload1=3.3,
    Rload2=10.0,
    Rload3=6.8,
    Vin=46.0,
    VF=1.1,
)

REL_TOL = Parameters(
    L=0.50,
    RL=0.4,
    C=0.50,
    RC=0.50,
    Rdson=0.5,
    Rload1=0.3,
    Rload2=0.3,
    Rload3=0.3,
    Vin=0.3,
    VF=0.3,
)


# %% [markdown]
# ## Consider residual correlation

# %%
# get data noise values

# ADC noise is 1 LSB, so we can assume that the data noise is 1 LSB
# 5 noise is 5 LSB, 10 noise is 10 LSB

# A least significant bit (LSB) is the smallest unit of data in a digital system and is calculated as:
#         LSB_i = I_FS / (2**12 - 1)  # assuming 12-bit ADC
#         LSB_v = V_FS / (2**12 - 1)  # assuming 12-bit ADC

# where I_FS is the full-scale current, set to 10 A and V_FS is the full-scale voltage set to 30 V (see 03_inspect_noisy_data.ipynb).

# Then the noise level is calculated as:
#         noise_level_i = noise_level * LSB_i  # normalize noise level to LSB
#         noise_level_v = noise_level * LSB_v  # normalize noise level to LSB

lsb_i = 10 / (2**12 - 1)  # 10 A full-scale current
lsb_v = 30 / (2**12 - 1)  # 30 V full-scale voltage

# i and v noise levels should probably be considered separately:

sigma_noise_ADC_i = 1 * lsb_i  # 1 LSB noise
sigma_noise_5_i = 5 * lsb_i  # 5 LSB noise
sigma_noise_10_i = 10 * lsb_i  # 10 LSB noise

sigma_noise_ADC_v = 1 * lsb_v  # 1 LSB noise
sigma_noise_5_v = 5 * lsb_v  # 5 LSB noise
sigma_noise_10_v = 10 * lsb_v  # 10 LSB noise

noise_power_ADC_i = sigma_noise_ADC_i**2
noise_power_5_i = sigma_noise_5_i**2
noise_power_10_i = sigma_noise_10_i**2

noise_power_ADC_v = sigma_noise_ADC_v**2
noise_power_5_v = sigma_noise_5_v**2
noise_power_10_v = sigma_noise_10_v**2


# %%
import torch
from torch.autograd.functional import jacobian


# load the transient data as unified numpy arrays
def load_data_to_model(meas: Measurement, initial_guess_params: Parameters):
    """Load the data from a Measurement object and return the model."""
    # load the transient data as unified numpy arrays
    X, y = meas.data
    s1, s2, s3 = list(
        map(lambda x: x - 1, meas.transient_lengths)
    )  # subtract 1 since we use the previous time step as input
    lb, ub = X.min(0), X.max(0)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    # Model
    model = BuckParamEstimator(lb, ub, s1, s2, s3, initial_guess_params).to(device)
    return X_t, y_t, model


def estimate_avg_Jacobian(model, X, y, max_samples=10) -> torch.Tensor:
    model.eval()
    X = X.detach()
    y = y.detach()

    jacobians_fw = []
    jacobians_bk = []

    for x_n, y_n in zip(X[:max_samples], y[:max_samples]):

        x_n = x_n.detach()
        x_n.requires_grad_(False)

        # Extract D and dt from x_n (they’re fixed)
        D = x_n[2].unsqueeze(0)
        dt = x_n[3].unsqueeze(0)

        def f_noise_inputs_bk(y):
            x_input = x_n.unsqueeze(0)             # shape (1, 4)
            y_input = y.unsqueeze(0)           # shape (1, 2)
            i_n, v_n, _, _ = model(x_input, y_input)
            return torch.cat([i_n, v_n], dim=1).squeeze()

        def f_noisy_inputs_fw(i_v):
            x_full = torch.cat([i_v, D, dt], dim=0).unsqueeze(0)
            y_input = y_n.unsqueeze(0)
            _, _, i_np1, v_np1 = model(x_full, y_input)
            return torch.cat([i_np1, v_np1], dim=1).squeeze()

        i_v_input = x_n[:2].clone().detach().requires_grad_(True)
        y_input = y_n.clone().detach().requires_grad_(True)
        J_fw = jacobian(f_noisy_inputs_fw, i_v_input)  # shape (2, 2)
        J_bk = jacobian(f_noise_inputs_bk, y_input)  # shape (2, 2)
        jacobians_fw.append(J_fw)
        jacobians_bk.append(J_bk)

    return torch.stack(jacobians_fw).mean(0), torch.stack(jacobians_bk).mean(0)  # average over all jacobians


# jacobians_fw = {}
# jacobians_bk = {}

# # Loop through all groups and estimate the Frobenius norm for each group

# # Load and assemble dataset
# db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
# h5filename = "buck_converter_Shuai_processed.h5"

# GROUP_NUMBER_DICT = {
#     0: "ideal",
#     1: "ADC_error",
#     3: "5 noise",
#     4: "10 noise",
# }


# for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
#     if "Sync" in group_name:
#         # Skip the Sync Error group for now
#         continue
#     print(f"Loading group {group_number}: {group_name}")
#     # Load the data from the hdf5 file
#     io = LoaderH5(db_dir, h5filename)
#     io.load(group_name)

#     X_t, y_t, model = load_data_to_model(io.M, initial_guess_params=NOMINAL)

#     print(f"Estimating Jacobian for group {group_name}...")
#     jac_fw, jac_bw = estimate_avg_Jacobian(model, X_t, y_t, max_samples=300)
#     jacobians_fw[group_name] = jac_fw
#     jacobians_bk[group_name] = jac_bw
#     print(f"Forward Jacobain for group {group_name} ({jac_fw.shape}): {jac_fw}")
#     print(f"Backward Jacobain for group {group_name} ({jac_bw.shape}): {jac_bw}")

# # average the Frobenius norms across all groups
# J_fw_av = torch.stack(list(jacobians_fw.values())).mean(0)
# J_bw_av = torch.stack(list(jacobians_bk.values())).mean(0)

# print(f"Average fw jacobian norm across all groups: {J_fw_av}")
# print(f"Average bw jacobian norm across all groups: {J_bw_av}")

# # %%
# # in theory, the Jacobian should be symmetric and the backward Jacobian should be the inverse of the forward Jacobian
# # Check the symmetry and inverse properties

# sym_error = torch.norm(J_fw_av @ J_bw_av - torch.eye(2))
# inv_error = torch.norm(J_bw_av - torch.linalg.inv(J_fw_av))
# print(f"Symmetry error ‖J_fw @ J_bw - I‖: {sym_error:.3e}")
# print(f"Inverse error ‖J_bw - inv(J_fw)‖: {inv_error:.3e}")


# J_av = 0.5 * (J_fw_av + torch.linalg.inv(J_bw_av))
# print(f"By using both forward and backward Jacobians, we can estimate the average Jacobian as:\n {J_av}")

# save the jacobian
jacobian_dir = Path.cwd() / "RESULTS" / "Jacobains" / "N0"


# load the jacobian
J_av = torch.load(jacobian_dir / "jacobian.pt")

# %%
# Now we have the jacobian we can calculate the Sigma matrix for the forward and backward optimization model
def noise_power_to_sigma(
    noise_power_i: float,
    noise_power_v: float,
    ) -> torch.Tensor:
    
    return torch.tensor(
        [
            [noise_power_i, 0.0],
            [0.0, noise_power_v]
        ],
        dtype=torch.float32,
    )

def estimate_sigma_fw_bw(
    sigma_x: torch.Tensor,
    J: torch.Tensor,
    calculate_diag_terms: bool = True,
): 
    def helper_diag_terms(
        sigma_x: torch.Tensor,
        J: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper function to compute the diagonal terms of the Sigma matrix.
        """
        return J @ sigma_x @ J.T + sigma_x
    
    j_inv: torch.Tensor = torch.linalg.inv(J)

    if calculate_diag_terms:
        sig_12 = - J @ sigma_x - sigma_x @ j_inv.T
        sig_21 = - j_inv @ sigma_x - sigma_x @ J.T
        
    else:
        sig_12 = torch.zeros((2, 2), device=sigma_x.device)
        sig_21 = torch.zeros((2, 2), device=sigma_x.device)
    
    # build the 4x4 Sigma matrix
    Sigma = torch.zeros((4, 4), device=sigma_x.device)
    Sigma[:2, :2] = helper_diag_terms(sigma_x, J)  # top-left
    Sigma[:2, 2:] = sig_12  # top-right
    Sigma[2:, :2] = sig_21  # bottom-left
    Sigma[2:, 2:] = helper_diag_terms(sigma_x, j_inv)
    
    return Sigma

def ensure_positive_definite(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Ensure the Sigma matrix is positive definite by adding a small value to the diagonal.
    """
    return Sigma + torch.eye(Sigma.shape[0], device=Sigma.device) * eps

def ensure_positive_definite_eigenvalues(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Ensure the Sigma matrix is positive definite by adjusting its eigenvalues.
    """
    eigvals = torch.linalg.eigvals(Sigma)
    min_eigval = torch.min(torch.real(eigvals))
    if min_eigval < eps:
        Sigma += (eps - min_eigval) * torch.eye(Sigma.shape[0], device=Sigma.device)
    return Sigma


sigma_adc = estimate_sigma_fw_bw(
    noise_power_to_sigma(
        noise_power_ADC_i,
        noise_power_ADC_v,
    ),
    J_av,
)
sigma_5 = estimate_sigma_fw_bw(
    noise_power_to_sigma(
        noise_power_5_i,
        noise_power_5_v,
    ),
    J_av,
)
sigma_10 = estimate_sigma_fw_bw(
    noise_power_to_sigma(
        noise_power_10_i,
        noise_power_10_v,
    ),
    J_av,
)


print(f"Sigma matrix for ADC noise:\n{sigma_adc}")
print(f"Sigma matrix for 5 LSB noise:\n{sigma_5}")
print(f"Sigma matrix for 10 LSB noise:\n{sigma_10}")

eps = 1e-2

sigma_adc = ensure_positive_definite_eigenvalues(sigma_adc, eps)
sigma_5 = ensure_positive_definite_eigenvalues(sigma_5, eps)
sigma_10 = ensure_positive_definite_eigenvalues(sigma_10, eps)

print("______________________")
print("Ensured positive definiteness of Sigma matrices:")
print(f"Sigma matrix for ADC noise:\n{sigma_adc}")
print(f"Sigma matrix for 5 LSB noise:\n{sigma_5}")
print(f"Sigma matrix for 10 LSB noise:\n{sigma_10}")


# investigate positive definiteness of the Sigma matrices
def is_positive_definite(matrix: torch.Tensor) -> bool:
    """Check if a matrix is positive definite."""
    return torch.all(torch.real(torch.linalg.eigvals(matrix)) > 0)


def distance_to_pd(Sigma):
    eigvals = torch.linalg.eigvals(Sigma)
    neg_part = torch.clamp(torch.real(eigvals), max=0).abs()
    return neg_part.sum().item()


print(f"Distance to positive definiteness for ADC noise: {distance_to_pd(sigma_adc):.3e}")
print(f"Distance to positive definiteness for 5 LSB noise: {distance_to_pd(sigma_5):.3e}")
print(f"Distance to positive definiteness for 10 LSB noise: {distance_to_pd(sigma_10):.3e}")

# %%
torch.linalg.eigvals(sigma_adc), torch.linalg.eigvals(sigma_5), torch.linalg.eigvals(sigma_10)

# %%
# now train with loss r Sigma^{-1} r^T loss
def fw_bw_loss_with_sigma(
    pred_np1: torch.Tensor,
    pred_n: torch.Tensor,
    observations_np1: torch.Tensor,
    observations_n: torch.Tensor,
    precision_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the loss with the Sigma matrix.
    """
    # Compute the residuals for both predictions
    residual_np1 = pred_np1 - observations_np1  # shape (batch_size, 2)
    residual_n = pred_n - observations_n  # shape (batch_size, 2)

    # residual vector
    r = torch.cat((residual_np1, residual_n), dim=1)  # shape (batch_size, 4)

    # Compute the loss as r^T Sigma^{-1} r
    mahal = torch.einsum("Bi, ij, Bj -> B", r, precision_matrix, r)  # shape (batch_size,)
    return 0.5 * mahal.sum()  # scalar


def fw_bw_loss_whitened(
    pred_np1: torch.Tensor,
    pred_n: torch.Tensor,
    observations_np1: torch.Tensor,
    observations_n: torch.Tensor,
    L_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Compute r^T Σ^{-1} r via Cholesky whitening: r -> z = L^{-1} r.
    """
    residual_np1 = pred_np1 - observations_np1  # shape (batch_size, 2)
    residual_n = pred_n - observations_n  # shape (batch_size, 2)
    r = torch.cat((residual_np1, residual_n), dim=1)  # (B, 4)

    z = torch.matmul(r, L_inv.T)  # whitening: (B, 4)
    return 0.5 * (z**2).sum()

def train_from_measurement_file(
    meas: Measurement,
    sigma: torch.Tensor,
    initial_guess_params: Parameters,
    savename: str = "saved_run",
    db_dir: Path = ".",
    lr_adam: float = 1e-3,
    adam_epochs: int = 15_000,
    kfac_epochs: int = 15_000,
    device: str = "cpu",
):
    # ---------------------------- data ---------------------------------
    X, y = meas.data
    s1, s2, s3 = [L - 1 for L in meas.transient_lengths]  # shift by −1
    lb, ub = X.min(0), X.max(0)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    # ---------------------------- model --------------------------------
    model = BuckParamEstimator(lb, ub, s1, s2, s3, initial_guess_params).to(device)

    # ----------------------- build Σ−½  (whitening) --------------------
    sigma_diag = sigma.clone()
    sigma_diag[:2, 2:] = 0.0
    sigma_diag[2:, :2] = 0.0

    def chol_inv(mat: torch.Tensor) -> torch.Tensor:
        """return (LLᵀ)⁻¹ᐟ² = L⁻ᵀ   where LLᵀ = mat (add jitter if needed)"""
        eps = 1e-6
        mat = mat + eps * torch.eye(mat.size(0), device=mat.device)
        L = torch.linalg.cholesky(mat)
        return torch.cholesky_inverse(L)  # same as L⁻ᵀ · L⁻¹

    L_inv_diag = chol_inv(sigma_diag).to(device)  # (4,4)
    L_inv_full = chol_inv(sigma).to(device)  # (4,4)

    # ----------------------- bookkeeping --------------------------------
    history_loss, history_params = [], []
    best_loss, best_iter = float("inf"), -1
    best_params = model.get_estimates()

    # ----------------------- stage 1: Adam ------------------------------
    adam_optim = torch.optim.Adam(model.parameters(), lr=lr_adam)

    for it in range(1, adam_epochs+1):
        adam_optim.zero_grad()

        i_n, v_n, i_np1, v_np1 = model(X_t, y_t)
        pred_np1 = torch.cat((i_np1, v_np1), dim=1)
        pred_n = torch.cat((i_n, v_n), dim=1)
        loss = fw_bw_loss_whitened(
            pred_np1,
            pred_n,
            observations_np1=y_t,
            observations_n=X_t[:, :2],
            L_inv=L_inv_diag,
        )
        loss.backward()
        adam_optim.step()

        if it % 1000 == 0:
            est = model.get_estimates()
            print(
                f"[Adam] iter {it:>6}, loss {loss.item():.4e}, "
                f"L={est.L:.3e} RL={est.RL:.3e} C={est.C:.3e} ..."
            )
            history_loss.append(loss.item())
            history_params.append(est)
            if loss.item() < best_loss:
                best_loss, best_iter = loss.item(), it


    # -------------------------- stage 2: K-FAC ----------------------------
    # Replace LBFGS with SGD + K-FAC second-order preconditioning

    sgd_optim = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    kfac_precond = KFACPreconditioner(
        model,
        factor_update_steps=10,
        inv_update_steps=100,
        damping=1e-3,
        factor_decay=0.95,
        kl_clip=0.001,
        lr=0.01,
    )


    nan_abort = True  # whether to raise on NaNs

    # ------------------------------------------------------------------
    # K-FAC training loop
    # ------------------------------------------------------------------
    for it in range(kfac_epochs):
        sgd_optim.zero_grad()

        # forward + loss
        i_n, v_n, i_np1, v_np1 = model(X_t, y_t)
        pred_np1 = torch.cat((i_np1, v_np1), dim=1)
        pred_n = torch.cat((i_n, v_n), dim=1)

        loss = fw_bw_loss_whitened(
            pred_np1,
            pred_n,
            observations_np1=y_t,
            observations_n=X_t[:, :2],
            L_inv=L_inv_full,
        )

        if not torch.isfinite(loss):
            msg = "[K-FAC] Non-finite loss encountered"
            if nan_abort:
                raise RuntimeError(msg)
            print(msg)
            break

        loss.backward()

        # Optional: detect NaN gradient early
        if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
            msg = "[K-FAC] NaN gradient detected"
            if nan_abort:
                raise RuntimeError(msg)
            print(msg)
            break

        # K-FAC preconditioning + SGD step
        kfac_precond.step()
    
        
        sgd_optim.step()

        # Parameter sanity check
        if any(not torch.isfinite(p.data).all() for p in model.parameters()):
            print("[K-FAC] NaN detected in params after update")
            break

        # Logging
        if (it + 1) % 1 == 0:
            global_it = adam_epochs + it + 1
            est = model.get_estimates()
            print(
                f"[K-FAC] iter {global_it:6}, loss {loss.item():.4e}, "
                f"L={est.L:.3e} RL={est.RL:.3e} C={est.C:.3e} ..."
            )
            history_loss.append(loss.item())
            history_params.append(est)
            if loss.item() < best_loss:
                best_loss, best_iter = loss.item(), global_it
                best_params = est

    # ----------------------- save history -------------------------------
    training_run = TrainingRun.from_histories(history_loss, history_params)
    db_dir.mkdir(parents=True, exist_ok=True)
    if not savename.endswith(".csv"):
        savename += ".csv"
    training_run.save_to_csv(db_dir / savename)

    print(f"Finished: best loss {best_loss:.4e} at iter {best_iter}")
    return best_params


def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(123)
device = "cpu"

# Load and assemble dataset
db_dir = Path(r"C:/Users/JC28LS/OneDrive - Aalborg Universitet/Desktop/Work/Databases")
h5filename = "buck_converter_Shuai_processed.h5"

GROUP_NUMBER_DICT = {
    0: "ideal",
    1: "ADC_error",
    3: "5 noise",
    4: "10 noise",
}


sigma_ideal = torch.eye(4, dtype=torch.float32) * 1e-9

# GROUP_NUMBER_TO_SIGMA = {
#     0: sigma_ideal,
#     1: sigma_ideal,
#     3: sigma_ideal,
#     4: sigma_ideal,
# }

GROUP_NUMBER_TO_SIGMA = {
    0: sigma_ideal,
    1: sigma_adc,
    3: sigma_5,
    4: sigma_10,
}


lr = 1e-3
epochs_adam = 20_000
epochs_kfac = 20_000
patience = 5000
device = "cpu"  # or "cuda" if you have a GPU
lr_reduction_factor = 0.5


out_dir = (
    Path.cwd().parent
    / "RESULTS"
    / "Testing"
    / "forward_vs_forward&backward"
    / "f&b_with_LBFGS"
)

noisy_measurements = {}
for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    if idx == 0:
        # Skip the ideal group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    # Load the data from the hdf5 file
    io = LoaderH5(db_dir, h5filename)
    io.load(group_name)

    # Store the measurement in a dictionary
    noisy_measurements[group_name] = io.M

    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")

    # Train the model on the noisy measurement
    best_params = train_from_measurement_file(
        io.M,
        sigma=GROUP_NUMBER_TO_SIGMA[group_number],
        initial_guess_params=NOMINAL,
        db_dir=out_dir,
        savename=f"noisy_run_{group_name}.csv",
        lr_adam=lr,
        adam_epochs=5000,
        kfac_epochs=epochs_kfac,
    )

    #print the best parameters
    print(f"Best parameters for {group_name}:")
    print(f"L: {best_params.L:.3e}, RL: {best_params.RL:.3e}, C: {best_params.C:.3e}, "
          f"RC: {best_params.RC:.3e}, Rdson: {best_params.Rdson:.3e}, "
          f"Rload1: {best_params.Rload1:.3e}, Rload2: {best_params.Rload2:.3e}, "
          f"Rload3: {best_params.Rload3:.3e}, Vin: {best_params.Vin:.3e}, VF: {best_params.VF:.3e}")
    print(f"{'-'*50}\n")


print("Done")
# %%


# %%


# %%


# %%


# %% [markdown]
# Certainly. Here is a schematic and technical write-up for your notes, focused on the failure modes of Adam when transitioning from a diagonal to full covariance loss:
#
# ---
#
# ### Why the Loss Increases After Switching to the Full Covariance Matrix
#
# We consider the Mahalanobis loss on the concatenated forward and backward residuals:
#
# $$
# \mathcal{L}(\theta) = \tfrac{1}{2} \sum_{n=1}^N r_n^\top \Sigma_r^{-1} r_n,
# $$
#
# where $r_n \in \mathbb{R}^4$ is the concatenated residual at timestep $n$ and $\Sigma_r \in \mathbb{R}^{4 \times 4} $ is a fixed residual covariance matrix built from the noise structure and average Jacobian.
#
# Empirically, we observe that switching from a simplified block-diagonal $\Sigma_r$ to a full matrix (including off-diagonal blocks) causes a sudden **increase in loss and parameter error**, despite using the same optimizer (e.g., Adam) and reducing the learning rate.
#
# ---
#
# #### 1. Loss Surface Becomes Anisotropic
#
# The full $\Sigma_r^{-1}$ introduces *directionally coupled curvature* into the loss. The gradient becomes:
#
# $$
# \nabla_\theta \mathcal{L} = \sum_{n=1}^N \left( \frac{\partial r_n^\top}{\partial \theta} \Sigma_r^{-1} r_n \right),
# $$
#
# which couples all four residual components. The Hessian $\nabla^2_\theta \mathcal{L}$ inherits the anisotropy of $\Sigma_r^{-1}$, leading to large variations in curvature across directions.
#
# Let $\lambda_{\max}$, $\lambda_{\min}$ be the extremal eigenvalues of the Hessian. Then the condition number
#
# $$
# \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \gg 1
# $$
#
# implies a narrow curved valley in parameter space, with steep and flat directions.
#
# ---
#
# #### 2. Adam Cannot Align with Curvature
#
# Adam updates parameters as:
#
# $$
# \theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon},
# $$
#
# where $m_t$ and $v_t$ are exponential moving averages of the gradient and its square, **computed coordinate-wise**.
#
# This per-parameter rescaling assumes the loss is axis-aligned. But under a full $\Sigma_r^{-1}$, steepest descent directions are **rotated** in parameter space. Adam’s update rule cannot rotate to follow these directions, and may thus:
#
# * Overshoot in stiff directions (high curvature → exploding gradient);
# * Undershoot in soft directions (low curvature → vanishing step);
# * Get trapped in curved valleys and oscillate.
#
# ---
#
# #### 3. Instability from Cross-Terms
#
# When residuals $r^{\mathrm{fw}}$ and $r^{\mathrm{bw}}$ are not yet well matched, the off-diagonal blocks of $\Sigma_r^{-1}$ can amplify cancellation or reinforce mismatch.
#
# Let:
#
# $$
# r = \begin{bmatrix}
# r^{\mathrm{fw}} \\
# r^{\mathrm{bw}}
# \end{bmatrix}, \quad
# \Sigma_r^{-1} =
# \begin{bmatrix}
# A & B \\
# B^\top & C
# \end{bmatrix}.
# $$
#
# Then:
#
# $$
# r^\top \Sigma_r^{-1} r = r^{\mathrm{fw} \top} A r^{\mathrm{fw}} + r^{\mathrm{bw} \top} C r^{\mathrm{bw}} + 2 r^{\mathrm{fw} \top} B r^{\mathrm{bw}}.
# $$
#
# If $B$ is not aligned with the current residuals, the **cross-term** can dominate and increase even when both forward and backward residuals decrease individually.
#
# ---
#
# #### 4. Loss Mismatch Due to Fixed $\Sigma_r$
#
# $\Sigma_r$ is computed using a fixed Jacobian $J$, often estimated from initial conditions. But the model evolves during training, and the residual function $r(\theta)$ becomes misaligned with the structure of $\Sigma_r$. This introduces a loss-function–model mismatch:
#
# $$
# \text{Optimizing } \mathcal{L}(\theta) = \tfrac{1}{2} r(\theta)^\top \Sigma_r^{-1} r(\theta)
# \quad \text{where } \Sigma_r \not\approx \text{Cov}(r(\theta)).
# $$
#
# The true Mahalanobis geometry drifts, but the optimizer continues to follow outdated curvature.
#
# ---
#
# #### 5. Practical Remedies
#
# * **Warm-up phase**: Train with block-diagonal or diagonal $\Sigma_r$ for stability.
# * **Gradual interpolation**: Blend the loss via
#
#   $$
#   \mathcal{L}_\alpha = (1-\alpha) \mathcal{L}_{\text{diag}} + \alpha \mathcal{L}_{\text{full}}, \quad \alpha \nearrow 1,
#   $$
#
#   over a large number of steps.
# * **Whitening**: Compute Cholesky $L = \text{chol}(\Sigma_r)$ and use whitened residuals
#
#   $$
#   z = L^{-1} r, \quad \mathcal{L} = \tfrac{1}{2} \|z\|^2,
#   $$
#
#   to improve conditioning.
# * **Switch optimizer**: Use LBFGS or second-order methods once in the full-covariance regime.
# * **Laplace fit post-training**: Train entirely with stable $\Sigma_r^{\text{diag}}$, then compute a posterior via Laplace approximation using the full $\Sigma_r$ and a frozen MAP estimate.
#
# ---
#
# #### Summary
#
# Even when the full covariance $\Sigma_r$ is mathematically correct, introducing it too early or too abruptly can destabilize optimization. Adam, with its diagonal adaptation, is not suited for strongly anisotropic or misaligned loss landscapes. Stable training requires either a transition phase or a separation between optimization (MAP estimation) and posterior inference (Laplace or Bayesian updates).
#

# %%


# %%


# %%
from pinn_buck.plot_utils import (
    plot_tracked_parameters,
    plot_final_percentage_error,
    plot_final_percentage_error_multi,
)

# loop through all CSV files in the directory
fb_outdir = (
    Path.cwd().parent / "RESULTS" / "Testing" / "forward_vs_forward&backward" / "forward&backward"
)
fbc_outdir = (
    Path.cwd().parent
    / "RESULTS"
    / "Testing"
    / "forward_vs_forward&backward"
    / "f&b_combined_residual"
)


csv_files = list(fbc_outdir.glob("*.csv"))
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

    label = "fbc_" + label
    if ii == 0:
        fig, ax = plot_tracked_parameters(
            df=tr,
            target=TRUE_PARAMS,
            label=label,
            color="black",
            figsize=(18, 10),
        )
        continue

    plot_tracked_parameters(df=tr, target=None, label=label, ax=ax, color=None)

runs_ordered_fbc = {GROUP_NUMBER_DICT[ii]: runs[GROUP_NUMBER_DICT[ii]] for ii in (0, 1, 3, 4)}


csv_files = list(fb_outdir.glob("*.csv"))
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

    label = "f&b_" + label
    plot_tracked_parameters(df=tr, target=None, label=label, ax=ax, color=None, linestyle="--")

runs_ordered_fb = {GROUP_NUMBER_DICT[ii]: runs[GROUP_NUMBER_DICT[ii]] for ii in (0, 1, 3, 4)}


fig, ax = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

plot_final_percentage_error_multi(
    runs=runs_ordered_fbc, target=TRUE_PARAMS, figsize=(14, 5), select_lowest_loss=False, ax=ax[0]
)
plot_final_percentage_error_multi(
    runs=runs_ordered_fb, target=TRUE_PARAMS, figsize=(14, 5), select_lowest_loss=False, ax=ax[1]
)

ax[0].set_title("Forward&Backward Combined Residual")
ax[1].set_title("Forward & Backward")

# %%
from typing import Callable, Union, Iterable
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from contextlib import contextmanager
import math

# let's define a function to convert relative tolerances to standard deviations
# using the log-normal distribution assumption.
# Previously, we assumed sigma = log(1 + rel_tol). This means we assume that the relative toleraces contain 1 standard deviation
# of the data. Although usually the relative tolerances are defined as 2 or 3 standard deviations, we will use 1 standard deviation
# since this is the worst case scenario.

def rel_tolerance_to_sigma(rel_tol: Parameters) -> Parameters:
    """Convert relative tolerances to standard deviations."""

    def _to_sigma(value: float) -> torch.Tensor:
        """Convert a relative tolerance to standard deviation."""
        return torch.log(torch.tensor(1 + value, dtype=torch.float32))

    return Parameters(
        L=_to_sigma(rel_tol.L),
        RL=_to_sigma(rel_tol.RL),
        C=_to_sigma(rel_tol.C),
        RC=_to_sigma(rel_tol.RC),
        Rdson=_to_sigma(rel_tol.Rdson),
        Rload1=_to_sigma(rel_tol.Rload1),
        Rload2=_to_sigma(rel_tol.Rload2),
        Rload3=_to_sigma(rel_tol.Rload3),
        Vin=_to_sigma(rel_tol.Vin),
        VF=_to_sigma(rel_tol.VF),
    )

# define the log-normal prior for the parameters assuming independent priors distrubuted according to the log-normal distribution.
# See the formula above.
def log_normal_prior(logparams: Parameters, nominal: Parameters, sigma: Parameters) -> torch.Tensor:
    """Return −log p(log z) assuming independent log-normal priors."""
    total = 0.0
    nominal_logparams = make_log_param(nominal)
    for name in Parameters._fields:
        proposed_value = getattr(logparams, name)
        mu = getattr(nominal_logparams, name)
        sig = getattr(sigma, name)
        total += ((proposed_value - mu) / sig) ** 2 / 2
    return total


def _parse_data_noise_to_sigma(data_noise: Union[float, Iterable, torch.Tensor]) -> torch.Tensor:
    """Parse data_noise and return the inverse covariance matrix Sigma_x_inv."""
    if isinstance(data_noise, float):
        a = data_noise
        b = data_noise
        return torch.diag(torch.tensor([a, b], dtype=torch.float32))
    elif isinstance(data_noise, torch.Tensor):
        if data_noise.shape != (2, 2):
            raise ValueError("If data_noise is a tensor, it must be 2x2.")
        return data_noise
    elif isinstance(data_noise, Iterable):
        data_noise = list(data_noise)
        if len(data_noise) != 2:
            raise ValueError("If data_noise is iterable, it must be of length 2.")
        a = data_noise[0]
        b = data_noise[1]
        return torch.diag(torch.tensor([a, b], dtype=torch.float32))
    else:
        raise TypeError("data_noise must be float, 2-tensor, or iterable of length 2.")


def likelihood_loss(preds, y_n, y_np1, Sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute −log likelihood with Mahalanobis norm using per-variable covariance.
    preds: tuple of (i_n, v_n, i_np1, v_np1)
    y_n, y_np1: true values at time steps n and n+1
    Sigma_x_inv: 2x2 inverse covariance matrix for [i, v]
    """
    i_n, v_n, i_np1, v_np1 = preds
    i0, v0 = y_n[:, 0:1], y_n[:, 1:2]
    i1, v1 = y_np1[:, 0:1], y_np1[:, 1:2]

    # Residuals: shape [N, 2]
    res_0 = torch.cat([i_n - i0, v_n - v0], dim=1)
    res_1 = torch.cat([i_np1 - i1, v_np1 - v1], dim=1)

    # Stack both sets of residuals: shape [2N, 2]
    residuals = torch.cat([res_0, res_1], dim=0)

    # Cholensky approach
    L = torch.linalg.cholesky(Sigma)                    # (2,2), lower-tri
    z = torch.linalg.solve_triangular(L, residuals.T, upper=False).T
    return 0.5 * z.pow(2).sum()


def likelihood_loss_triplets(
    preds: torch.Tensor, targets: torch.Tensor, Sigma: torch.Tensor  # (N,4)  # (N,4)  # (2,2)
) -> torch.Tensor:
    """
    Negative log-likelihood for Δ-residuals under
    r_Δ ∼ 𝒩(0, Σ),    Σ = Sigma (2×2).
    """
    # --- split columns ------------------------------------------------
    i_np1_pred, v_np1_pred, i_n_pred, v_n_pred = preds.T  # (N,)
    i_np1_true, v_np1_true, i_n_true, v_n_true = targets.T  # (N,)

    # --- build increments --------------------------------------------
    delta_pred = torch.stack([i_np1_pred - i_n_pred, v_np1_pred - v_n_pred], dim=1)  # (N,2)
    delta_obs = torch.stack([i_np1_true - i_n_true, v_np1_true - v_n_true], dim=1)  # (N,2)

    residuals = delta_pred - delta_obs  # (N,2)

    # --- Mahalanobis term via whitening ------------------------------
    L = torch.linalg.cholesky(Sigma)  # (2,2)
    z = torch.linalg.solve_triangular(L, residuals.T, upper=False).T  # (N,2)
    nll = 0.5 * z.pow(2).sum()  # scalar
    return nll


# define the loss function for MAP estimation that combines the L2 loss and the log-normal prior.
def make_map_loss(
    nominal: Parameters, sigma: Parameters, residual_covariance: Union[float, Iterable, torch.Tensor] = 1.0
) -> Callable:

    residual_covariance = _parse_data_noise_to_sigma(residual_covariance)
    def _loss(model, preds, targets):
        ll = likelihood_loss_triplets(preds, targets, residual_covariance)
        prior = log_normal_prior(model.logparams, nominal, sigma) 
        return ll + prior

    return _loss

# %%
from dataclasses import dataclass

# for simplicity let's define a dataclass for the training configurations
@dataclass
class AdamOptTrainingConfigs:
    savename: str = "saved_run"
    out_dir: Path = Path(".")
    lr: float = 1e-3
    epochs: int = 30_000
    epochs_lbfgs: int = 1500
    device: str = "cpu"
    patience: int = 5000
    lr_reduction_factor: float = 0.5

from typing import Callable, Dict, List, Any
from functools import partial


class TrainerTriplets:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optim_cfg: "AdamOptTrainingConfigs",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        # expects (preds, targets)
        self.optim_cfg = optim_cfg
        self.device = device
        self.loss_fn = loss_fn
        # history for plotting / CSV export
        self.history: Dict[str, List[Any]] = {"loss": [], "params": [], "lr": []}

    # -----------------------------------------------------------------
    def _record(self, loss_val: float):
        est = self.model.get_estimates()
        self.history["loss"].append(loss_val)
        self.history["params"].append(est)
        # LR from the first param-group (Adam & LBFGS both expose it)
        self.history["lr"].append(self.opt.param_groups[0]["lr"])

    # -----------------------------------------------------------------
    def fit(self, X: torch.Tensor, epochs_adam: int = 20_000, epochs_lbfgs: int = 500):

        X = X.to(self.device)

        # ------------- Adam phase ------------------------------------
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.optim_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=self.optim_cfg.lr_reduction_factor,
            patience=self.optim_cfg.patience,
        )

        best_loss = float("inf")

        for it in range(epochs_adam):
            self.opt.zero_grad()
            preds, targets = self.model(X)  # <- new API
            loss = self.loss_fn(self.model, preds, targets)
            loss.backward()
            self.opt.step()
            scheduler.step(loss.item())

            if (it % 1000) == 0:
                self._record(loss.item())
                best_loss = min(best_loss, loss.item())

                est = self.model.get_estimates()
                grad_norm = (
                    torch.cat(
                        [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]
                    )
                    .norm()
                    .item()
                )

                print(
                    f"[Adam {it:>6}] "
                    f"loss={loss.item():.3e}, grad‖={grad_norm:.3e}, "
                    f"L={est.L:.2e}, C={est.C:.2e}, "
                    f"Rload1={est.Rload1:.2e}, Rload2={est.Rload2:.2e}, "
                    f"Rload3={est.Rload3:.2e}"
                )

        print("Adam finished.  Best loss:", best_loss)

        # ------------- LBFGS phase -----------------------------------
        print("Starting LBFGS optimisation …")
        self.opt = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=epochs_lbfgs,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-8,
        )

        def closure():
            self.opt.zero_grad()
            preds, targets = self.model(X)
            loss = self.loss_fn(self.model, preds, targets)
            loss.backward()
            return loss

        self.opt.step(closure)
        final_loss = closure().item()
        self._record(final_loss)
        print("LBFGS finished.  Final loss:", final_loss)
        # print the final parameter estimations
        print("Final parameter estimates:")
        est = self.model.get_estimates()
        print(
            f"L={est.L:.2e}, C={est.C:.2e}, "
            f"Rload1={est.Rload1:.2e}, Rload2={est.Rload2:.2e}, "
            f"Rload3={est.Rload3:.2e},"
        )

        # ------------- save history ----------------------------------
        run = TrainingRun.from_histories(
            loss_history=self.history["loss"],
            param_history=self.history["params"],
        )
        out_dir = self.optim_cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_name = (
            self.optim_cfg.savename
            if str(self.optim_cfg.savename).endswith(".csv")
            else f"{self.optim_cfg.savename}.csv"
        )
        run.save_to_csv(out_dir / csv_name)

        return self.model

# %%
# get data noise values

# ADC noise is 1 LSB, so we can assume that the data noise is 1 LSB
# 5 noise is 5 LSB, 10 noise is 10 LSB

# A least significant bit (LSB) is the smallest unit of data in a digital system and is calculated as:
#         LSB_i = I_FS / (2**12 - 1)  # assuming 12-bit ADC
#         LSB_v = V_FS / (2**12 - 1)  # assuming 12-bit ADC

# where I_FS is the full-scale current, set to 10 A and V_FS is the full-scale voltage set to 30 V (see 03_inspect_noisy_data.ipynb).

# Then the noise level is calculated as:
#         noise_level_i = noise_level * LSB_i  # normalize noise level to LSB
#         noise_level_v = noise_level * LSB_v  # normalize noise level to LSB

lsb_i = 10 / (2**12 - 1)  # 10 A full-scale current
lsb_v = 30 / (2**12 - 1)  # 30 V full-scale voltage

# i and v noise levels should probably be considered separately:

sigma_noise_ADC_i = 1 * lsb_i  # 1 LSB noise
sigma_noise_5_i = 5 * lsb_i  # 5 LSB noise
sigma_noise_10_i = 10 * lsb_i  # 10 LSB noise

sigma_noise_ADC_v = 1 * lsb_v  # 1 LSB noise
sigma_noise_5_v = 5 * lsb_v  # 5 LSB noise
sigma_noise_10_v = 10 * lsb_v  # 10 LSB noise

noise_power_ADC_i = sigma_noise_ADC_i**2
noise_power_5_i = sigma_noise_5_i**2
noise_power_10_i = sigma_noise_10_i**2

noise_power_ADC_v = sigma_noise_ADC_v**2
noise_power_5_v = sigma_noise_5_v**2
noise_power_10_v = sigma_noise_10_v**2

# %% [markdown]
# Now let's estimate the Frobenius norm by using the autograd function included in pytorch.
#
# The best solution would be to **pretrain the model to be close to the correct physical quantities of the parameters**. But for simplicity we can use the nominal values.
#
# Moreover, the noise on i and the noise on v may be very different in magnitude. However, again for simplicity we calculate a single Frobenius norm for both.

# %%
from torch.autograd.functional import jacobian

# load the transient data as unified numpy arrays
def load_data_to_model(meas: Measurement, initial_guess_params: Parameters):
    """Load the data from a Measurement object and return the model."""
    # load the transient data as unified numpy arrays
    X, y = meas.data
    s1, s2, s3 = list(
        map(lambda x: x - 1, meas.transient_lengths)
    )  # subtract 1 since we use the previous time step as input
    lb, ub = X.min(0), X.max(0)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    # Model
    model = BuckParamEstimator(lb, ub, s1, s2, s3, initial_guess_params).to(device)
    return X_t, y_t, model


def estimate_avg_Jacobian(model, X, y, max_samples=10)-> torch.Tensor:
    model.eval()
    X = X.detach()
    y = y.detach()

    jacobians = []

    for x_n, y_n in zip(X[:max_samples], y[:max_samples]):

        x_n = x_n.detach()
        x_n.requires_grad_(False)

        # Extract D and dt from x_n (they’re fixed)
        D = x_n[2].unsqueeze(0)
        dt = x_n[3].unsqueeze(0)

        # Define function of ONLY the noisy inputs: i, v
        def f_noisy_inputs(i_v):
            x_full = torch.cat([i_v, D, dt], dim=0).unsqueeze(0)
            y_input = y_n.unsqueeze(0)
            i_pred, v_pred = model(x_full, y_input)[2:]
            return torch.cat([i_pred, v_pred], dim=1).squeeze()  # shape (2,)

        i_v_input = x_n[:2].clone().detach().requires_grad_(True)
        J = jacobian(f_noisy_inputs, i_v_input)  # shape (2, 2)
        jacobians.append(J)
    return torch.stack(jacobians).mean(0)  # average over all jacobians


jacobians = {}
# Loop through all groups and estimate the Frobenius norm for each group

for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    # Load the data from the hdf5 file
    io = LoaderH5(db_dir, h5filename)
    io.load(group_name)

    X_t, y_t, model = load_data_to_model(io.M, initial_guess_params=NOMINAL)

    print(f"Estimating Jacobian for group {group_name}...")
    jac = estimate_avg_Jacobian(model, X_t, y_t, max_samples=300)
    jacobians[group_name] = jac
    print(f"Jacobain for group {group_name} ({jac.shape}): {jac}")

# average the Frobenius norms across all groups
J_av = torch.stack(list(jacobians.values())).mean(0)
print(f"Average Frobenius norm across all groups: {jacobian}")

# %%
# what is the error in the Frobenius norm estimation, with respect to the true parameter values?
for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    # Load the data from the hdf5 file
    io = LoaderH5(db_dir, h5filename)
    io.load(group_name)

    X_t, y_t, model_tr = load_data_to_model(io.M, initial_guess_params=TRUE_PARAMS)

    print(f"Estimating Jacobian for group {group_name}...")
    jac = estimate_avg_Jacobian(model_tr, X_t, y_t, max_samples=300)
    jacobians[group_name] = jac
    print(f"Jacobain for group {group_name} ({jac.shape}): {jac}")

# %%
def estimate_avg_frobenius_norm(model, X, y, max_samples=10):
    model.eval()
    X = X.detach()
    y = y.detach()

    norms_squared = []

    for x_n, y_n in zip(X[:max_samples], y[:max_samples]):

        x_n = x_n.detach()
        x_n.requires_grad_(False)

        # Extract D and dt from x_n (they’re fixed)
        D = x_n[2].unsqueeze(0)
        dt = x_n[3].unsqueeze(0)

        # Define function of ONLY the noisy inputs: i, v
        def f_noisy_inputs(i_v):
            x_full = torch.cat([i_v, D, dt], dim=0).unsqueeze(0)
            y_input = y_n.unsqueeze(0)
            i_pred, v_pred = model(x_full, y_input)[2:]
            return torch.cat([i_pred, v_pred], dim=1).squeeze()  # shape (2,)

        i_v_input = x_n[:2].clone().detach().requires_grad_(True)
        J = jacobian(f_noisy_inputs, i_v_input)  # shape (2, 2)
        frob_norm_sq = torch.norm(J, p="fro") ** 2
        norms_squared.append(frob_norm_sq.item())

    return sum(norms_squared) / len(norms_squared)


# compare with the Frobenius norm used previously
frob_sq_direct = torch.norm(J_av, p="fro").pow(2)
frob_sq_avg = estimate_avg_frobenius_norm(model, X_t, y_t, max_samples=300)

print(f"Frobenius norm squared (direct): {frob_sq_direct:.3e}")
print(f"Frobenius norm squared (average): {frob_sq_avg:.3e}")

# %% [markdown]
# We can see that the values of the Frobenius norm with the true parameters are quite close to the one obtained with the nominal parameters!
#
# Now we can rescale the noise powers using the formula:
#
#    $$
#    \boxed{
#    \operatorname{Var}[r_\Delta]
#      \approx (J+I)\Sigma_x(J+I)^{\!\top} + J\Sigma_x J^{\!\top} + \Sigma_x.
#    }
#    $$

# %%
# rescale the noise powers
Sigma_x_ADC = torch.tensor([[noise_power_ADC_i, 0], [0, noise_power_ADC_v]], dtype=torch.float32)
Sigma_x_5 = torch.tensor([[noise_power_5_i, 0], [0, noise_power_5_v]], dtype=torch.float32)
Sigma_x_10 = torch.tensor([[noise_power_10_i, 0], [0, noise_power_10_v]], dtype=torch.float32)

# transform noise power to the covariance matrix of the residuals
def transform_noise_to_residual_covariance(noise_power: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Transform noise power to the covariance matrix of the residuals."""
    # J is the Jacobian, noise_power is a 2x2 diagonal matrix
    return (J + torch.eye(2)) @ noise_power @ (J + torch.eye(2)).T + J @ noise_power @ J.T + noise_power

# Propagate the noise through the Jacobian to get the rescaled noise power
Sigma_tot_ADC = transform_noise_to_residual_covariance(Sigma_x_ADC, J_av)
Sigma_tot_5 = transform_noise_to_residual_covariance(Sigma_x_5, J_av)
Sigma_tot_10 = transform_noise_to_residual_covariance(Sigma_x_10, J_av)

print(f"Sigma_tot_ADC:\n{Sigma_tot_ADC}")
print(f"Sigma_tot_5:\n{Sigma_tot_5}")
print(f"Sigma_tot_10:\n{Sigma_tot_10}")


# %%
from pinn_buck.model.model_param_estimator import BuckParamEstimatorTriplets

out_dir = Path.cwd().parent / "RESULTS" / "Bayesian" / "Adam_MAP"

run_configs = AdamOptTrainingConfigs(
    savename="adam_run.csv",
    out_dir=out_dir,
    lr=lr,
    epochs=30000,  # 30k epochs for Adam
    epochs_lbfgs=10000,
    device=device,
    patience=patience,
    lr_reduction_factor=lr_reduction_factor,
)


# load the transient data as unified numpy arrays
def load_data_to_model(meas: Measurement, initial_guess_params: Parameters):
    """Load the data from a Measurement object and return the model."""
    # load the transient data as unified numpy arrays
    X, y = meas.data
    s1, s2, s3 = list(
        map(lambda x: x - 1, meas.transient_lengths)
    )  # subtract 1 since we use the previous time step as input
    lb, ub = X.min(0), X.max(0)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    # Model
    model = BuckParamEstimatorTriplets(lb, ub, s1, s2, s3, initial_guess_params).to(device)
    return X_t, y_t, model

noise_power_dict = {
    0: 1e-9,  # ideal
    1: Sigma_tot_ADC,  # ADC error
    3: Sigma_tot_5,  # 5 noise
    4: Sigma_tot_10,  # 10 noise
}


noisy_measurements = {}
trained_models = {}
inverse = False

for idx, (group_number, group_name) in enumerate(GROUP_NUMBER_DICT.items()):
    if "Sync" in group_name:
        # Skip the Sync Error group for now
        continue
    print(f"Loading group {group_number}: {group_name}")
    # Load the data from the hdf5 file
    io = LoaderH5(db_dir, h5filename)
    io.load(group_name)

    # Store the measurement in a dictionary
    noisy_measurements[group_name] = io.M

    run_configs.savename = f"noisy_run_{group_name}.csv"

    print(f"\n{'-'*50}")
    print(f"{idx}) Training with {group_name} data")

    # Train the model on the noisy measurement
    X, y, model = load_data_to_model(
        meas=io.M,
        initial_guess_params=NOMINAL,
    )

    prior_info = {
        "nominal": NOMINAL,
        "sigma": rel_tolerance_to_sigma(REL_TOL),
    }


    trainer = TrainerTriplets(
        model=model,
        loss_fn=make_map_loss(
            **prior_info,
            residual_covariance=noise_power_dict[group_number],  # use the noise power for the group
        ),
        optim_cfg=run_configs,
        device=device,
    )

    trainer.fit(
        X=X,
        epochs_adam=run_configs.epochs,
        epochs_lbfgs=run_configs.epochs_lbfgs,
    )
    inverse = True  # inverse is False only for the ideal case, so we set it to True for the rest of the groups
    trained_models[group_name] = trainer.model
    print("\n \n \n")

# %%
from typing import Dict
import matplotlib.pyplot as plt


# loop through all CSV files in the directory
csv_files = list(out_dir.glob("*.csv"))
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
runs_ordered: Dict[str, TrainingRun] = {
    GROUP_NUMBER_DICT[ii]: runs[GROUP_NUMBER_DICT[ii]] for ii in (0, 1, 3, 4)
}

# Plotting
param_names = Parameters._fields
ncols = 2
nrows = int(np.ceil(len(param_names) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))
axes = axes.flatten()


for i, name in enumerate(param_names):
    nominal = getattr(NOMINAL, name)
    rel_tol = getattr(REL_TOL, name)
    true_val = getattr(TRUE_PARAMS, name)
    estimations_for_different_runs: Dict[str, Parameters] = {
        label: getattr(run.best_parameters, name) for label, run in runs_ordered.items()
    }

    # log-normal parameters
    sigma = np.log(1 + rel_tol)
    mu = np.log(nominal)

    dist = lognorm(s=sigma, scale=np.exp(mu))

    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
    pdf = dist.pdf(x)

    ax = axes[i]
    ax.plot(x, pdf, label=f"{name} prior")
    ax.axvline(true_val, color="red", linestyle="--", label="TRUE")

    # plot the nominal value
    ax.axvline(nominal, color="green", linestyle="--", label="Nominal")

    for label, est in estimations_for_different_runs.items():
        ax.axvline(est, linestyle=":", label=f"{label} estimate")
    ax.set_title(name)
    ax.set_yticks([])
    ax.legend()

    if name == "L":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
        ax.set_xlabel("[mH]")
    if name == "C":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
        ax.set_xlabel("[μF]")

fig.suptitle("Log-Normal Priors (linear space) with TRUE values", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.95)

# %% [markdown]
# ## Laplace Approximation
#
# The **Laplace approximation** is a method for approximating a complex posterior distribution with a Gaussian centered at the maximum a posteriori (MAP) estimate. It assumes that the log-posterior is approximately quadratic near its peak, allowing the posterior to be approximated as:
#
# $$
# p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta_{\text{MAP}}, \Sigma), \quad \Sigma^{-1} = \nabla^2_{\theta} [-\log p(\theta \mid \mathcal{D})] \big|_{\theta = \theta_{\text{MAP}}}
# $$
#
# This approximation provides an estimate of uncertainty around the MAP point by evaluating the curvature (i.e., the Hessian) of the negative log-posterior.
#
# We use the Laplace approximation to fit a Gaussian (and optionally a log-normal) distribution to the posterior over the model parameters, enabling uncertainty quantification around the point estimate obtained by optimization.

# %%
from torch.autograd.functional import hessian
from torch.func import functional_call
from scipy.stats import norm, lognorm
import numpy as np
import torch
from torch import nn
from torch.autograd.functional import hessian
from torch.func import functional_call
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

from scipy.stats import norm, lognorm  # --- utilities

# -----------------------------------------------------------------
# user-supplied helpers
#   Parameters, make_log_param, reverse_log_param
#   log_normal_prior, rel_tolerance_to_sigma
#   likelihood_loss_triplets, _parse_data_noise_to_sigma
# must already be imported
# -----------------------------------------------------------------


# -----------------------------------------------------------------#
#   Container for the posterior                                   #
# -----------------------------------------------------------------#
@dataclass
class LaplacePosterior:
    theta_log: torch.Tensor  # MAP in log-space
    Sigma_log: torch.Tensor  # covariance in log-space
    theta_phys: torch.Tensor  # MAP in physical units
    Sigma_phys: torch.Tensor  # covariance in physical units


# -----------------------------------------------------------------#
#   LaplaceFitter class                                            #
# -----------------------------------------------------------------#
class LaplaceFitter:
    """
    Compute a Laplace (Gaussian) approximation to the posterior of a
    BuckParamEstimatorTriplets model.
    """

    # ------------- construction -----------------------------------
    def __init__(
        self,
        model: nn.Module,
        X: torch.Tensor,  # full dataset (concatenated runs)
        noise_power: torch.Tensor,  # 2×2 Σ_Δ   (already propagated)
        NOMINAL: Parameters,
        REL_TOL: Parameters,
        damping: float = 1e-6,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.X = X.to(device)
        self.Sigma_delta = noise_power.to(device)
        self.NOMINAL = NOMINAL
        self.REL_TOL = REL_TOL
        self.damping = damping
        self.device = device

        # pre-compute scale σ for the log-normal prior
        self.prior_sigma = rel_tolerance_to_sigma(REL_TOL)

    # ------------- helper: flatten current log-parameters ---------
    def _flat_logparams(self) -> torch.Tensor:
        """Return  (P,)  vector, requires_grad=True."""
        vec = torch.cat([p.detach().clone().view(1) for p in self.model.logparams]).to(self.device)
        vec.requires_grad_(True)
        return vec

    # ------------- build closure L(θ) ------------------------------
    def _posterior_loss_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns f(theta_vec) that:
            1) rewrites model parameters,
            2) runs the triplet forward,
            3) computes −log posterior.
        """
        param_keys = [name for name, _ in self.model.named_parameters()]
        assert len(param_keys) == len(Parameters._fields), "param count mismatch"

        def loss(theta_vec: torch.Tensor) -> torch.Tensor:
            # split flat θ into individual tensors with correct shapes
            split = []
            offset = 0
            for name, p0 in self.model.named_parameters():
                n = p0.numel()
                split.append(theta_vec[offset : offset + n].view_as(p0))
                offset += n

            state_dict = {k: v for k, v in zip(param_keys, split)}  # new θ
            preds, targets = functional_call(self.model, state_dict, (self.X,))

            # likelihood
            ll = likelihood_loss_triplets(preds, targets, self.Sigma_delta)

            # prior (independent log-normal)
            logparams = Parameters(*split)
            prior = log_normal_prior(logparams, self.NOMINAL, self.prior_sigma)

            return ll + prior

        return loss

    # ------------- main entry -------------------------------------
    def fit(self) -> LaplacePosterior:
        theta_map = self._flat_logparams()
        loss_fn = self._posterior_loss_fn()

        # ----- compute MAP gradient once (optional sanity check) ---
        loss_map = loss_fn(theta_map)
        loss_map.backward()

        # ----- Hessian --------------------------------------------
        H = hessian(loss_fn, theta_map)
        H = (H + H.T) * 0.5  # symmetrise

        I = torch.eye(H.shape[0], device=self.device)
        Sigma_log = torch.linalg.inv(H + self.damping * I)

        # ----- convert to physical units ---------------------------
        theta_phys = torch.tensor(
            [getattr(self.model.get_estimates(), n) for n in Parameters._fields],
            device=self.device,
        )
        J = torch.diag(theta_phys)  # ∂θ_phys/∂θ_log = diag(θ_phys)
        Sigma_phys = J @ Sigma_log @ J.T

        return LaplacePosterior(
            theta_log=theta_map.detach(),
            Sigma_log=Sigma_log,
            theta_phys=theta_phys,
            Sigma_phys=Sigma_phys,
        )

    # -------- convenience static helpers --------------------------
    @staticmethod
    def build_gaussian_approx(mean: np.ndarray, cov: np.ndarray):
        std = np.sqrt(np.diag(cov))
        return [norm(loc=m, scale=s) for m, s in zip(mean, std)]

    @staticmethod
    def build_lognormal_approx(mu_log: np.ndarray, sigma_log: np.ndarray):
        return [lognorm(s=s, scale=np.exp(m)) for m, s in zip(mu_log, sigma_log)]

    @staticmethod
    def print_parameter_uncertainty(theta_phys, Sigma_phys):
        std_phys = torch.sqrt(torch.diag(Sigma_phys))
        for i, name in enumerate(Parameters._fields):
            mean = theta_phys[i].item()
            std = std_phys[i].item()
            pct = 100.0 * std / mean
            print(f"{name:8s}: {mean:.3e} ± {std:.1e} ({pct:.2f} %)")

# %%
label_noise_dict = {
    "ideal": torch.tensor([[1e-9, 0.0], [0.0, 1e-9]], dtype=torch.float32),  # no noise
    "ADC_error": Sigma_tot_ADC,
    "5 noise": Sigma_tot_5,
    "10 noise": Sigma_tot_10
}

lfits = {}

for label, model in trained_models.items():
    noise_power = label_noise_dict[label]

    # Fit Laplace posterior using the new class
    laplace = LaplaceFitter(
        model=model,
        X=X,
        noise_power=noise_power,
        NOMINAL=NOMINAL,
        REL_TOL=REL_TOL,
        damping=1e-6,
        device="cpu",  # or "cuda" if using GPU
    )
    lfit = laplace.fit()

    # Compute Gaussian and LogNormal approximations
    gaussians = LaplaceFitter.build_gaussian_approx(
        mean=lfit.theta_phys.cpu().numpy(),
        cov=lfit.Sigma_phys.cpu().numpy()
    )

    lognormals = LaplaceFitter.build_lognormal_approx(
        mu_log=lfit.theta_log.cpu().numpy(),
        sigma_log=np.sqrt(torch.diag(lfit.Sigma_log).cpu().numpy())
    )

    # Print and store
    print(f"\nParameter estimates for {label}:")
    LaplaceFitter.print_parameter_uncertainty(lfit.theta_phys, lfit.Sigma_phys)
    lfits[label] = lfit


# %%
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
import numpy as np


def plot_laplace_posteriors(
    lfit: LaplacePosterior,
    NOMINAL: Parameters,
    REL_TOL: Parameters,
    TRUE_PARAMS: Parameters,
    _SCALE: dict,
    runs_ordered: dict,
):
    """
    Plot Laplace approximations (Gaussian + log-normal) against log-normal prior for each parameter.
    """
    param_names = Parameters._fields
    ncols = 2
    nrows = int(np.ceil(len(param_names) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 14))
    axes = axes.flatten()

    # Convert tensors to numpy
    theta_map_np = lfit.theta_phys.cpu().numpy()
    std_map_np = np.sqrt(torch.diag(lfit.Sigma_phys).cpu().numpy())
    mu_log_np = lfit.theta_log.cpu().detach().numpy()
    sigma_log_np = np.sqrt(torch.diag(lfit.Sigma_log).cpu().numpy())

    for i, name in enumerate(param_names):
        nominal = getattr(NOMINAL, name)
        rel_tol = getattr(REL_TOL, name)
        true_val = getattr(TRUE_PARAMS, name)
        mu_post = theta_map_np[i]
        std_post = std_map_np[i]

        # Prior distribution
        sigma_prior = np.log(1 + rel_tol)
        mu_prior = np.log(nominal)
        prior_dist = lognorm(s=sigma_prior, scale=np.exp(mu_prior))
        x_prior = np.linspace(prior_dist.ppf(0.001), prior_dist.ppf(0.999), 500)
        y_prior = prior_dist.pdf(x_prior)

        # Posterior (Gaussian)
        x_post_gauss = np.linspace(mu_post - 4 * std_post, mu_post + 4 * std_post, 500)
        y_post_gauss = norm(loc=mu_post, scale=std_post).pdf(x_post_gauss)

        # Posterior (log-normal, Laplace)
        mu_log = mu_log_np[i] - np.log(_SCALE[name])  # adjust for scale
        sigma_log = sigma_log_np[i]
        post_dist = lognorm(s=sigma_log, scale=np.exp(mu_log))
        x_post = np.linspace(post_dist.ppf(0.001), post_dist.ppf(0.999), 500)
        y_post = post_dist.pdf(x_post)

        ax = axes[i]
        ax.plot(x_prior, y_prior, label="Prior (log-normal)", color="blue", linewidth=1)
        ax.plot(
            x_post_gauss,
            y_post_gauss,
            label="Laplace Posterior (Gaussian)",
            color="orange",
            linewidth=1,
        )
        ax.plot(x_post, y_post, label="Laplace Posterior (log-normal)", color="black", linewidth=2)

        # markers
        ax.axvline(true_val, color="red", linestyle="--", label="TRUE")
        ax.axvline(nominal, color="green", linestyle="--", label="Nominal")
        ax.axvline(mu_post, color="purple", linestyle="-.", label="MAP Estimate")
        
        # point estimates from other runs
        # for label, run in runs_ordered.items():
        #     est = getattr(run.best_parameters, name)
        #     ax.axvline(est, linestyle=":", label=f"{label} estimate")

        ax.set_title(name)
        ax.set_yticks([])

        # Format axis labels
        if name == "L":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
            ax.set_xlabel("[mH]")
        elif name == "C":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
            ax.set_xlabel("[μF]")

        ax.legend(fontsize="x-small", loc="upper right")

    fig.suptitle("Prior (log-normal) and Laplace Posterior for Each Parameter", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()

# %%
label = "ADC_error"  # change this to "ideal", "5 noise", or "10 noise" as needed

lfit = lfits[label]
plot_laplace_posteriors(
    lfit = lfit,
    NOMINAL=NOMINAL,
    REL_TOL=REL_TOL,
    TRUE_PARAMS=TRUE_PARAMS,
    _SCALE=_SCALE,
    runs_ordered=runs_ordered,
)

# %%
def plot_single_laplace_posterior(
    param_name: str,
    lfit: LaplacePosterior,
    _SCALE: dict,
    ax: plt.Axes,
    label: str,
    style: str = "log-normal",  # or "gaussian"
    color: str = None,
    plot_prior: bool = False,
    NOMINAL: Parameters = None,
    REL_TOL: Parameters = None,
    TRUE_PARAMS: Parameters = None,
    show_map_marker: bool = True,
    marker_kwargs: dict = None,
):
    """
    Plot a single Laplace posterior distribution and optionally the prior,
    with a square marker at the MAP estimate placed on the PDF curve.

    Args:
        param_name: name of the parameter
        theta_phys: MAP in physical units
        Sigma_phys: covariance in physical units
        theta_log: MAP in log space
        Sigma_log: covariance in log space
        _SCALE: scaling dictionary
        ax: matplotlib axis to draw on
        label: label for the posterior
        style: 'log-normal' or 'gaussian'
        color: curve and marker color
        plot_prior: show log-normal prior
        NOMINAL, REL_TOL: required if plot_prior is True
        show_map_marker: draw square on posterior at MAP
        marker_kwargs: customization for square marker
    """
    param_names = Parameters._fields
    idx = param_names.index(param_name)

    mu_post = lfit.theta_phys[idx].item()
    std_post = float(torch.sqrt(lfit.Sigma_phys[idx, idx]))

    if plot_prior:
        if NOMINAL is None or REL_TOL is None:
            raise ValueError("NOMINAL and REL_TOL must be provided to plot the prior.")
        nominal = getattr(NOMINAL, param_name)
        rel_tol = getattr(REL_TOL, param_name)
        sigma_prior = np.log(1 + rel_tol)
        mu_prior = np.log(nominal)
        prior_dist = lognorm(s=sigma_prior, scale=np.exp(mu_prior))
        x_prior = np.linspace(prior_dist.ppf(0.001), prior_dist.ppf(0.999), 500)
        y_prior = prior_dist.pdf(x_prior)
        ax.plot(x_prior, y_prior, label="Prior (log-normal)", color="black", linewidth=2)
        if show_map_marker:
            y_nominal_prior = prior_dist.pdf(nominal)
            default_marker_kwargs = {"marker": "s", "color": "black", "s": 30, "zorder": 5}
            ax.scatter([nominal], [y_nominal_prior], **default_marker_kwargs)

    if style == "gaussian":
        x = np.linspace(mu_post - 4 * std_post, mu_post + 4 * std_post, 500)
        y = norm(loc=mu_post, scale=std_post).pdf(x)
        (line,) = ax.plot(x, y, label=label, color=color, linewidth=1)
        if show_map_marker:
            y_map = norm(loc=mu_post, scale=std_post).pdf(mu_post)
    elif style == "log-normal":
        mu_log = lfit.theta_log[idx].item() - np.log(_SCALE[param_name])
        sigma_log = float(torch.sqrt(lfit.Sigma_log[idx, idx]))
        dist = lognorm(s=sigma_log, scale=np.exp(mu_log))
        x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
        y = dist.pdf(x)
        (line,) = ax.plot(x, y, label=label, color=color, linewidth=1)
        if show_map_marker:
            y_map = dist.pdf(mu_post)
    else:
        raise ValueError(f"Unknown style '{style}'; use 'gaussian' or 'log-normal'.")

    line_color = line.get_color() if color is None else color

    # Add square marker on the curve at the MAP
    if show_map_marker:
        default_marker_kwargs = {"marker": "s", "color": line_color, "s": 30, "zorder": 5}
        if marker_kwargs:
            default_marker_kwargs.update(marker_kwargs)
        ax.scatter([mu_post], [y_map], **default_marker_kwargs)

    if TRUE_PARAMS is not None:
        true_val = getattr(TRUE_PARAMS, param_name)
        ax.axvline(true_val, color="red", linestyle="--", label="TRUE", linewidth=1)

    ax.set_title(param_name)
    ax.set_yticks([])

    if param_name == "L":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
        ax.set_xlabel("[mH]")
    elif param_name == "C":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
        ax.set_xlabel("[μF]")

def plot_all_laplace_posteriors_grid(
    lfits: dict[str, LaplacePosterior],
    NOMINAL: Parameters,
    REL_TOL: Parameters,
    _SCALE: dict,
    TRUE_PARAMS: Parameters = None,
    skip_labels: set = {"ideal"},
    style: str = "log-normal",
):
    """
    Plot posterior PDFs from Laplace approximation for all parameters in a grid.

    Args:
        trained_models: dict[label] -> trained model
        X, y: input data tensors
        NOMINAL, REL_TOL: prior hyperparams
        _SCALE: scaling dict
        label_noise_dict: dict[label] -> data_noise_power
        TRUE_PARAMS: optional, used to show true value marker
        skip_labels: set of labels to skip (e.g. 'ideal')
        style: 'log-normal' or 'gaussian'
    """
    param_names = Parameters._fields
    ncols = 2
    nrows = int(np.ceil(len(param_names) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 14))
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Only show prior and true param once
        show_prior = True
        true_params = TRUE_PARAMS

        for label, fit in lfits.items():
            if label in skip_labels:
                continue
            plot_single_laplace_posterior(
                param_name=param_name,
                lfit=fit,
                _SCALE=_SCALE,
                ax=ax,
                label=label,
                style=style,
                plot_prior=show_prior,
                NOMINAL=NOMINAL,
                REL_TOL=REL_TOL,
                show_map_marker=True,
                marker_kwargs={"label": None},  # avoid duplicate legend entry
                TRUE_PARAMS=true_params,
            )
            if show_prior:
                show_prior = False  # only plot once
            true_params = None

        # Add true value line (if provided)
        if TRUE_PARAMS:
            true_val = getattr(TRUE_PARAMS, param_name)
            ax.axvline(true_val, color="red", linestyle="--", label="TRUE")

        ax.set_title(param_name)
        ax.legend(fontsize="x-small", loc="upper right")

    fig.suptitle("Laplace Posteriors with Varying Data Noise", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()

# %%
parameter_name = "VF"


fig, ax = plt.subplots(figsize=(6, 4))
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

plot_prior = True
true_params = TRUE_PARAMS
for label, lfit in lfits.items():

    plot_single_laplace_posterior(
        param_name=parameter_name,
        lfit = lfit,
        _SCALE=_SCALE,
        ax=ax,
        label=label,
        style="log-normal",
        plot_prior=plot_prior,
        NOMINAL=NOMINAL,
        REL_TOL=REL_TOL,
        TRUE_PARAMS=true_params,
    )

    ax.legend(title="Data with Different Noises:")
    ax.set_title("Laplace Posterior and Prior for " + parameter_name)
    plt.tight_layout()
    plot_prior = False  # only plot prior once
    true_params = None  # don't plot TRUE_PARAMS again

# %%
plot_all_laplace_posteriors_grid(
    lfits=lfits,
    NOMINAL=NOMINAL,
    REL_TOL=REL_TOL,
    _SCALE=_SCALE,
    TRUE_PARAMS=TRUE_PARAMS,
)

# %% [markdown]
# Great! We have obtained posterior distributions on the estimations, which depend on the nose of the input data.
#
#
# ## Limitations
#
# #### 0. Choice of Priors
# The choice of priors is done by inspection
#
# ## Remedies and Next Steps
#
# For points 1 and 3 we can think of different approaches to solve this problem.
#
# 1. Rather than simply considering $x_{\text{obs}}$, estimate the true x considering this as a latent variable
#
# 2. Model the noise scale explicitly, learning $\sigma^2$ as a latent variable.
#
# 2. Use full Bayesian inference, these return better estimations of the posterior p(z | y, x):
#     + VI approach: should still be possible to use automatic-diff + Adam.
#     + HMC / NUTS: Since we only have 10 dimensions, there is no curse of dimensionality. Should be possible to draw **exact values from the posterior** without relying on surrogate models.
#
# ---
#
# Also, the prior effect is quite strong when the data is noisy. This may be correct and could be resolved with the previous points, but we can see what happens if we use uniform priors or **empirical Bayes**, which are chosen in a data-driven manner (maybe we can keep the nominal and only estimate $\sigma$ in this way).

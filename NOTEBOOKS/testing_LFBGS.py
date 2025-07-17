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

eps = 1e-4

sigma_adc = ensure_positive_definite(sigma_adc, eps)
sigma_5 = ensure_positive_definite(sigma_5, eps)
sigma_10 = ensure_positive_definite(sigma_10, eps)

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
    lbfgs_epochs: int = 15_000,
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

    # ----------------------- stage 2: LBFGS -----------------------------
    # ------------------------------------------------------------------
    #  LBFGS configuration
    # ------------------------------------------------------------------
    lbfgs_optim = torch.optim.LBFGS(
            model.parameters(),
            lr=1,            # works in your test
            max_iter=10,       # inner line-search iterations
            history_size=50,   # critical for stability
    )


    nan_abort  = True         # raise RuntimeError on NaN/Inf

    # ------------------------------------------------------------------
    #  Closure with finite checks
    # ------------------------------------------------------------------
    def closure():
        lbfgs_optim.zero_grad()

        i_n, v_n, i_np1, v_np1 = model(X_t, y_t)
        pred_np1 = torch.cat((i_np1, v_np1), dim=1)
        pred_n   = torch.cat((i_n,   v_n),   dim=1)

        loss_val = fw_bw_loss_whitened(
            pred_np1, pred_n,
            observations_np1=y_t,
            observations_n=X_t[:, :2],
            L_inv=L_inv_full,
        )

        # 1)  finite-loss check
        if not torch.isfinite(loss_val):
            message = "[LBFGS] Non-finite loss encountered"
            if nan_abort:
                raise RuntimeError(message)
            else:
                print(message)
                return loss_val

        loss_val.backward()
        return loss_val

    # ------------------------------------------------------------------
    #  LBFGS training loop
    # ------------------------------------------------------------------
    for it in range(lbfgs_epochs):
        try:
            loss = lbfgs_optim.step(closure)
        except RuntimeError as err:
            print(f"[LBFGS] Stopped at outer iter {it}: {err}")
            break

        # 3)  post-step parameter sanity
        with torch.no_grad():
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                print("[LBFGS] Non-finite parameter detected — aborting.")
                break

        if (it + 1) % 100 == 0:
            global_it = adam_epochs + it + 1
            est = model.get_estimates()
            print(
                f"[LBFGS] iter {global_it:6}, loss {loss.item():.4e}, "
                f"L={est.L:.3e}  RL={est.RL:.3e}  C={est.C:.3e} ..."
            )
            history_loss.append(loss.item())
            history_params.append(est)
            if loss.item() < best_loss:
                best_loss, best_iter = loss.item(), global_it
                best_params = model.get_estimates()


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
epochs_lbfgs = 20_000
patience = 5000
device = "cpu"  # or "cuda" if you have a GPU
lr_reduction_factor = 0.5


out_dir = (
    Path.cwd()
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
        lbfgs_epochs=500,
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

# %%
import argparse
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import h5py
import torch
import torch.nn as nn


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# %%
# nominal values

adam_epochs = 45000
adam_lr = 1e-3
lbfgs_epochs = 50000
L_n = 7.25*1e-4
RL_n = 0.314
C_n = 1.645 *1e-4
RC_n = 2.01 * 1e-1
Rdson_n = 0.221
Rload1_n = 3.1
Rload2_n = 10.2
Rload3_n = 6.1
Vin_n = 48
VF_n = 1.0


# %% [markdown]
# ## Let's Test The Accuracy of dt of Runge-Kutta

# %%
def predict_next_state(
    i0: float,
    v0: float,
    D: float,
    dt: float,
    L: float,
    RL: float,
    C: float,
    RC: float,
    Rdson: float,
    Rload: float,
    Vin: float,
    Vf: float,
    n_substeps: int = 100,
) -> Tuple[float, float]:
    """Predict next state using RK4 with optional sub-stepping."""
    h = dt / n_substeps  # smaller time step
    i, v = i0, v0

    def f(i, v):
        di = (-(RL + Rdson * D) * i - v + D * Vin - (1 - D) * Vf) / L
        dv = (Rload * i - v + C * RC * Rload * di) / (C * (RC + Rload))
        return di, dv

    for _ in range(n_substeps):
        k1_i, k1_v = f(i, v)
        k2_i, k2_v = f(i + 0.5 * h * k1_i, v + 0.5 * h * k1_v)
        k3_i, k3_v = f(i + 0.5 * h * k2_i, v + 0.5 * h * k2_v)
        k4_i, k4_v = f(i + h * k3_i, v + h * k3_v)

        i += (h / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        v += (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return i, v


import numpy as np
from pathlib import Path
from pinn_buck.io import TransientData

# Load data
db_dir = Path(r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases")
db_name = "buck_converter_Shuai_processed.h5"
tr1 = TransientData.from_h5(db_dir / db_name, "ideal", 1)

# Index to test
idx = 51
i0 = tr1.i[idx]
v0 = tr1.v[idx]
i1 = tr1.i[idx + 1]
v1 = tr1.v[idx + 1]
D = tr1.D[idx]
dt = tr1.dt[idx]

# Nominal parameters
i1_pred_ssteps, v1_pred_ssteps = predict_next_state(
    i0=i0,
    v0=v0,
    D=D,
    dt=dt,
    L=7.25e-4,
    RL=0.314,
    C=1.645e-4,
    RC=0.201,
    Rdson=0.221,
    Rload=3.1,
    Vin=48.0,
    Vf=1.0,
    n_substeps=1000,
)

# Predict without sub-steps
i1_pred, v1_pred = predict_next_state(
    i0=i0,
    v0=v0,
    D=D,
    dt=dt,
    L=L_n,
    RL=RL_n,
    C=C_n,
    RC=RC_n,
    Rdson=Rdson_n,
    Rload=Rload1_n,  # Using Rload1 as an example
    Vin=Vin_n,
    Vf=VF_n,
    n_substeps=1,  # No sub-steps
)

print("Without sub-steps:")
print(f"i1_pred = {i1_pred:.6f}, i1 = {i1:.6f}, Delta_i = {i1_pred - i1:.6e}")
print(f"v1_pred = {v1_pred:.6f}, v1 = {v1:.6f}, Delta_v = {v1_pred - v1:.6e}")



print("With 1000 sub-steps:")
print(f"i1_pred = {i1_pred_ssteps:.6f}, i1 = {i1:.6f}, Delta_i = {i1_pred_ssteps - i1:.6e}")
print(f"v1_pred = {v1_pred_ssteps:.6f}, v1 = {v1:.6f}, Delta_v = {v1_pred_ssteps - v1:.6e}")

# %% [markdown]
# **There is virtually no difference in the prediction error!**

# %%
## Create the datasets
from pinn_buck.io import TransientData

db_dir = Path(r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases")
db_name = "buck_converter_Shuai_processed.h5"

tr1 = TransientData.from_h5(db_dir / db_name, "ideal", 1)
tr2 = TransientData.from_h5(db_dir / db_name, "ideal", 2)
tr3 = TransientData.from_h5(db_dir / db_name, "ideal", 3)

i1_n = tr1.i[:-1]
v1_n = tr1.v[:-1]
i1_np1 = tr1.i[1:]
v1_np1 = tr1.v[1:]
D1 = tr1.D[:-1]
dt1 = tr1.dt[:-1]

i2_n = tr2.i[:-1]
v2_n = tr2.v[:-1]
i2_np1 = tr2.i[1:]
v2_np1 = tr2.v[1:]
D2 = tr2.D[:-1]
dt2 = tr2.dt[:-1]


i3_n = tr3.i[:-1]
v3_n = tr3.v[:-1]
i3_np1 = tr3.i[1:]
v3_np1 = tr3.v[1:]
D3 = tr3.D[:-1]
dt3 = tr3.dt[:-1]



i_n = np.concatenate((i1_n, i2_n, i3_n), axis=0)
i_np1 = np.concatenate((i1_np1, i2_np1, i3_np1), axis=0)
v_n = np.concatenate((v1_n, v2_n, v3_n), axis=0)
v_np1 = np.concatenate((v1_np1, v2_np1, v3_np1), axis=0)
D = np.concatenate((D1, D2, D3), axis=0)
dt = np.concatenate((dt1, dt2, dt3), axis=0)


X = np.stack([i_n, v_n, i_np1, v_np1, D, dt], axis=1)
Rload = np.concatenate(
    (np.full_like(i1_n, Rload1_n), np.full_like(i2_n, Rload2_n), np.full_like(i3_n, Rload3_n)), axis=0
)


class Normalizer:
    """Normalizer for the input data."""
    def __init__(self, X: np.ndarray):
        i_mean, i_std, v_mean, v_std, dt_mean, dt_std = self._get_means(X)
        # add dummy values for D
        self.mean = np.array([i_mean, v_mean, i_mean, v_mean, 0.0, dt_mean])
        self.std = np.array([i_std, v_std, i_std, v_std, 1, dt_std])
        
    
    def _get_means(self, X: np.ndarray) -> Tuple[float, float, float, float]:
        # i_all is i_n with concatenated the LAST VALUE of i_np1
        i_n_last = X[:, 0]  # i_n
        i_np1_last = X[-1, 2]
        i_all = np.concatenate((i_n_last, [i_np1_last]), axis=0)
        i_mean = i_all.mean()
        i_std = i_all.std()
        
        # v_all is v_n with concatenated the LAST VALUE of v_np1
        v_n_last = X[:, 1]
        v_np1_last = X[-1, 3]
        v_all = np.concatenate((v_n_last, [v_np1_last]), axis=0)
        v_mean = v_all.mean()
        v_std = v_all.std()
        
        dt_mean = X[:, 5].mean()
        dt_std = X[:, 5].std()
        return i_mean, i_std, v_mean, v_std, dt_mean, dt_std


    def normalize(self, X: np.ndarray) -> np.ndarray:
        # Normalize the input data X using the mean and std
        X_norm = (X - self.mean) / self.std
        return X_norm
    
    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        # Denormalize the input data X_norm using the mean and std
        X_denorm = X_norm * self.std + self.mean
        return X_denorm

normalizer = Normalizer(X)
X_norm = normalizer.normalize(X)    

# %%
## create simple feedforward neural network that estimates parameters
class ParamEstimator(nn.Module):
    """Predicts physical parameters [L, RL, C, RC, Rdson, Rload, Vin, Vf] from a single sample."""

    def __init__(self, input_dim: int, hidden_layers: List[int]):
        super().__init__()
        dims = [input_dim] + hidden_layers + [7]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            # last layer linear, others tanh
            if out_dim != 7:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [i_n, v_n, i_np1, v_np1, D, dt]
        return self.net(x)


def denorm_physical_params(
    L: torch.Tensor,
    RL: torch.Tensor,
    C: torch.Tensor,
    RC: torch.Tensor,
    Rdson: torch.Tensor,
    Vin: torch.Tensor,
    Vf: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Denormalize the physical parameters from their logarithmic form.
    The parameters are expected to be in the logarithmic scale.
    """
    L = torch.exp(L) * 1e-6  # assume the network gets the uH value
    RL = torch.exp(RL)
    C = torch.exp(C) * 1e-6  # assume the network gets the uF value
    RC = torch.exp(RC)
    Rdson = torch.exp(Rdson) * 1e-3  # the Rdson is quite small, so we assume it is in mOhm
    Vin = torch.exp(Vin)*10  # Vin is in V
    Vf = torch.exp(Vf)  # Vf is in V
    return L, RL, C, RC, Rdson, Vin, Vf


# --- Physics Forward RK4 ---
def physics_forward(
    x_n: torch.Tensor, params: torch.Tensor, normalizer: Normalizer, Rload: torch.Tensor
) -> torch.Tensor:
    """
    Given x_n = [i_n, v_n, i_np1, v_np1, D, dt] and predicted params,
    reconstruct x_np1_pred = [i_np1, v_np1].
    """
    # unnormalize the input data
    x_n = normalizer.denormalize(x_n)
    
    # unpack inputs
    i_n = x_n[:, 0:1]
    v_n = x_n[:, 1:2]
    D = x_n[:, 4:5]
    dt = x_n[:, 5:6]
    
    # unpack params
    L, RL, C, RC, Rdson, Vin, Vf = torch.split(params, 1, dim=1)

    # the model actually predicts the logarithm of the parameters:denormalize parameters

    L, RL, C, RC, Rdson, Vin, Vf = denorm_physical_params(
        L, RL, C, RC, Rdson, Vin, Vf
    )

    i_np1_pred, v_np1_pred = predict_next_state(
        i_n, v_n, D, dt, L, RL, C, RC, Rdson, Vin, Vf, Rload
    )

    return torch.cat([i_np1_pred, v_np1_pred], dim=1)


def predict_next_state(
    i_n: torch.Tensor,
    v_n: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    L: torch.Tensor,
    RL: torch.Tensor,
    C: torch.Tensor,
    RC: torch.Tensor,
    Rdson: torch.Tensor,
    Vin: torch.Tensor,
    Vf: torch.Tensor,
    Rload: torch.Tensor,
) -> torch.Tensor:
    """Predict the next state [i_np1, v_np1] using the RK4 method."""

    def f(i: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        di = -((D * Rdson + RL) * i + v - D * Vin + (1 - D) * Vf) / L
        dv = (C * RC * Rload * di + Rload * i - v) / (C * (RC + Rload))
        return di, dv

    # RK4 steps
    k1_i, k1_v = f(i_n, v_n)
    k2_i, k2_v = f(i_n + 0.5 * dt * k1_i, v_n + 0.5 * dt * k1_v)
    k3_i, k3_v = f(i_n + 0.5 * dt * k2_i, v_n + 0.5 * dt * k2_v)
    k4_i, k4_v = f(i_n + dt * k3_i, v_n + dt * k3_v)

    i_np1_pred = i_n + (dt / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
    v_np1_pred = v_n + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return i_np1_pred, v_np1_pred


# --- Loss ---
def compute_loss(x_np1_pred: torch.Tensor, x_np1_true: torch.Tensor) -> torch.Tensor:
    return torch.sum((x_np1_pred - x_np1_true) ** 2)


def criterion(model, X, normalizer):
    """
    Full-batch loss used by both Adam and L-BFGS.
    """
    params_pred = model(X)  # (N, 8)
    x_np1_pred = physics_forward(X, params_pred, normalizer)  # (N, 2)
    x_np1_true = X[:, 2:4]  # (N, 2)  ->  [i_{n+1}, v_{n+1}]
    return compute_loss(x_np1_pred, x_np1_true)



# %%

# train the model for 10 epochs

X_t = torch.tensor(X_norm, dtype=torch.float32)
Rload_t = torch.tensor(Rload, dtype=torch.float32).view(-1, 1)
model = ParamEstimator(input_dim=6, hidden_layers=[32, 32])
device = "cpu"

model.to(device)
X_t = X_t.to(device)
Rload_t = Rload_t.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

for epoch in range(int(10e4)):
    optimizer.zero_grad()
    
    # forward pass
    params_pred = model(X_t)
    x_n1_pred = physics_forward(X_t, params_pred, normalizer, Rload_t)
    loss = compute_loss(X_t[:, :2], x_n1_pred)
    
    # backward propagation
    loss.backward()
    optimizer.step()
    
    
    
    # get explicit predictions
    L, RL, C, RC, Rdson, Vin, Vf = torch.split(params_pred, 1, dim=1)
    
    # denormalize parameters and print the predicted values
    L, RL, C, RC, Rdson, Vin, Vf = denorm_physical_params(
        L, RL, C, RC, Rdson, Vin, Vf
    )
    
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch + 1}:")
        print(f"loss: {loss.item():.3e}, L: {L.mean().item():.6f} H, RL: {RL.mean().item():.6f} Ohm, C: {C.mean().item():.6f} F")
        print(f"RC: {RC.mean().item():.6f} Ohm, Rdson: {Rdson.mean().item():.6f} Ohm, Vin: {Vin.mean().item():.6f} V, Vf: {Vf.mean().item():.6f} V")

    


# %%
lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    max_iter=lbfgs_epochs, 
    tolerance_grad=1e-9,
    line_search_fn='strong_wolfe',
    )

def closure():
    lbfgs.zero_grad()
    loss = criterion(model, X_t, normalizer)
    loss.backward()
    return loss

print("Starting L‑BFGS optimisation … (this may take a while)")
lbfgs.step(closure)
print("L‑BFGS finished.")

# %%
L, RL, C, RC, Rdson, Rload, Vin, Vf = torch.split(model(X_t), 1, dim=1)

# Denormalize parameters
L, RL, C, RC, Rdson, Rload, Vin, Vf = denorm_physical_params(
    L, RL, C, RC, Rdson, Rload, Vin, Vf
)
print("Final parameters after L-BFGS:")

print(f"L: {L.mean().item():.6f} H vs {L_n} H")
print(f"RL: {RL.mean().item():.6f} Ohm vs {RL_n} Ohm")
print(f"C: {C.mean().item():.6f} F vs {C_n} F")
print(f"RC: {RC.mean().item():.6f} Ohm vs {RC_n} Ohm")
print(f"Rdson: {Rdson.mean().item():.6f} Ohm vs {Rdson_n} Ohm")
print(f"Rload: {Rload.mean().item():.6f} Ohm vs {Rload1} Ohm")
print(f"Vin: {Vin.mean().item():.6f} V vs {Vin_n} V")
print(f"Vf: {Vf.mean().item():.6f} V vs {VF_n} V")

# %%




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

# nominal values

adam_epochs = 45000
adam_lr = 1e-3
lbfgs_epochs = 50000
L = 7.25*1e-4
RL = 0.314
C = 1.645 *1e-4
RC = 2.01 * 1e-1
Rdson = 0.221
Rload1 = 3.1
Rload2 = 10.2
Rload3 = 6.1
Vin = 48
VF = 1.0


## create simple feedforward neural network that estimates parameters
class ParamEstimator(nn.Module):
    """Predicts physical parameters [L, RL, C, RC, Rdson, Rload, Vin, Vf] from a single sample."""
    def __init__(self, input_dim: int, hidden_layers: List[int]):
        super().__init__()
        dims = [input_dim] + hidden_layers + [8]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_dim, out_dim)
            layers.append(nn.Tanh())
            # Apply Xavier initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if out_dim != 8:
                # layers.append(nn.Tanh())
                nn.Linear(dims[-1], 8),
                nn.Softplus()  # ensures positivity
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [i_n, v_n, i_np1, v_np1, D, dt]
        return self.net(x)


# def denorm_physical_params(
#     L: torch.Tensor,
#     RL: torch.Tensor,
#     C: torch.Tensor,
#     RC: torch.Tensor,
#     Rdson: torch.Tensor,
#     Rload: torch.Tensor,
#     Vin: torch.Tensor,
#     Vf: torch.Tensor,
# ) -> Tuple[
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
#     torch.Tensor,
# ]:
#     """
#     Denormalize the physical parameters from their logarithmic form.
#     Clamp outputs to avoid numerical instability.
#     """
#     log_min, log_max = -5.0, 5.0  # reasonable log bounds

#     L = torch.exp(torch.clamp(L, log_min, log_max)) * 1e-6  # assume the network gets the uH value
#     RL = torch.exp(torch.clamp(RL, log_min, log_max))
#     C = torch.exp(torch.clamp(C, log_min, log_max)) * 1e-6  # assume the network gets the uF value
#     RC = torch.exp(torch.clamp(RC, log_min, log_max))
#     Rdson = (
#         torch.exp(torch.clamp(Rdson, log_min, log_max)) * 1e-3
#     )  # the Rdson is quite small, so we assume it is in mOhm
#     Rload = torch.exp(torch.clamp(Rload, log_min, log_max))
#     Vin = torch.exp(torch.clamp(Vin, log_min, log_max))
#     Vf = torch.exp(torch.clamp(Vf, log_min, log_max))
#     return L, RL, C, RC, Rdson, Rload, Vin, Vf


pysical_param_bounds = (1e-5, 1e-2)


def denorm_physical_params(
    L: torch.Tensor,
    RL: torch.Tensor,
    C: torch.Tensor,
    RC: torch.Tensor,
    Rdson: torch.Tensor,
    Rload: torch.Tensor,
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
    torch.Tensor,
]:
    """
    Denormalize the physical parameters from their logarithmic form.
    The parameters are expected to be in the logarithmic scale.
    """
    L = torch.clamp(L, *pysical_param_bounds) * 1e-6
    RL = torch.clamp(RL, *pysical_param_bounds)
    C = torch.clamp(C, *pysical_param_bounds) * 1e-6  # assume the network gets the uF value
    RC = torch.clamp(RC, *pysical_param_bounds)
    Rdson = torch.clamp(Rdson, *pysical_param_bounds) * 1e-1  # the Rdson is quite small
    Rload = torch.clamp(Rload, *pysical_param_bounds) * 10  # Rload is in Ohm
    Vin = torch.clamp(Vin, *pysical_param_bounds) * 10  # Vin is in V
    Vf = torch.clamp(Vf, *pysical_param_bounds)  # Vf is in V

    # since the network favors small values, let's artificially increase the prediction scale
    for param in [L, RL, C, RC, Rdson, Rload, Vin, Vf]:
        param *= 10.0
    return L, RL, C, RC, Rdson, Rload, Vin, Vf


# --- Physics Forward RK4 ---
def physics_forward(x_n: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Given x_n = [i_n, v_n, i_np1, v_np1, D, dt] and predicted params,
    reconstruct x_np1_pred = [i_np1, v_np1].
    """
    # unpack inputs
    i_n = x_n[:, 0:1]
    v_n = x_n[:, 1:2]
    D = x_n[:, 4:5]
    dt = x_n[:, 5:6]
    # unpack params
    L, RL, C, RC, Rdson, Rload, Vin, Vf = torch.split(params, 1, dim=1)
    
    # the model actually predicts the logarithm of the parameters
    
    # denormalize parameters 
    L, RL, C, RC, Rdson, Rload, Vin, Vf = denorm_physical_params(
        L, RL, C, RC, Rdson, Rload, Vin, Vf
    )
    
    # denormalize the currents, voltages, duty cycle, and time step with X_mean and X_std
    i_n = i_n * X_std[0, 0] + X_mean[0, 0]
    v_n = v_n * X_std[0, 1] + X_mean[0, 1]
    D = D * X_std[0, 4] + X_mean[0, 4]
    dt = dt * X_std[0, 5] + X_mean[0, 5]
    

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

    return torch.cat([i_np1_pred, v_np1_pred], dim=1)


# --- Loss ---
def compute_loss(x_np1_pred: torch.Tensor, x_np1_true: torch.Tensor) -> torch.Tensor:
    return torch.sum((x_np1_pred - x_np1_true) ** 2)

## Create the datasets
from pinn_buck.io import TransientData

db_dir = Path(r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases")
db_name = "buck_converter_Shuai_processed.h5"

tr1 = TransientData.from_h5(db_dir / db_name, "ideal", 1)

i_n = tr1.i[:-1]
v_n = tr1.v[:-1]
i_np1 = tr1.i[1:]
v_np1 = tr1.v[1:]
D = tr1.D[:-1]
dt = tr1.dt[:-1]


X = np.stack([i_n, v_n, i_np1, v_np1, D, dt], axis=1)


# normalize each column of X independently
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True)
X_norm = (X - X_mean) / X_std

Rload = Rload1 * np.ones_like(i_n)

# train the model for 10 epochs

X_t = torch.tensor(X_norm, dtype=torch.float32)
Rload_t = torch.tensor(Rload, dtype=torch.float32).view(-1, 1)
model = ParamEstimator(input_dim=6, hidden_layers=[32, 32])
device = "cpu"

model.to(device)
X_t = X_t.to(device)
Rload_t = Rload_t.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

for epoch in range(int(10e5)):
    optimizer.zero_grad()
    
    # forward pass
    params_pred = model(X_t)
    x_n1_pred = physics_forward(X_t, params_pred)
    loss = compute_loss(X_t[:, :2], x_n1_pred)
    
    # backward propagation
    loss.backward()
    optimizer.step()
    
    
    
    # get explicit predictions
    L, RL, C, RC, Rdson, Rload, Vin, Vf = torch.split(params_pred, 1, dim=1)
    
    # denormalize parameters and print the predicted values
    L, RL, C, RC, Rdson, Rload, Vin, Vf = denorm_physical_params(
        L, RL, C, RC, Rdson, Rload, Vin, Vf
    )
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch + 1}:")
        print(f"loss: {loss.item():.3e}, L: {L.mean().item():.6f} H, RL: {RL.mean().item():.6f} Ohm, C: {C.mean().item():.6f} F")
        print(f"RC: {RC.mean().item():.6f} Ohm, Rdson: {Rdson.mean().item():.6f} Ohm, Rload: {Rload.mean().item():.6f} Ohm, Vin: {Vin.mean().item():.6f} V, Vf: {Vf.mean().item():.6f} V")

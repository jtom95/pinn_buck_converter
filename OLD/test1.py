# %%
import argparse
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
import h5py

# %% [markdown]
# ## Define Buck Converter Properties

# %%
# Nominal component values (Table I in the paper)

L_n = 7.25 * 1e-4
RL_n = 0.314
C_n = 1.645 * 1e-4
RC_n = 0.201
Rdson_n = 0.221
Rload1 = 3.1
Rload2 = 10.2
Rload3 = 6.1
Vin_n = 48
VF_n = 1.0


L0 = 2.0 *1e-4
RL0 = 0.0039 
C0 = 0.412 *1e-4
RC0 = 0.159 
Rdson0 = 0.122
Rload10 = 1.22
Rload20 = 1.22
Rload30 = 1.22
Vin0 = 8.7 
VF = 0.1

# %% [markdown]
# ## Import Data from h5 database

# %%
from pinn_buck.h5_funcs import explore_h5

db_dir = Path(r"C:\Users\JC28LS\OneDrive - Aalborg Universitet\Desktop\Work\Databases")
db_name = "buck_converter_Shuai_processed.h5"


## Save the final quantities to a new HDF5 file
explore_h5(db_dir / db_name)

# %%
# load the data

from dataclasses import dataclass, field

@dataclass
class TransientData:
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    i: np.ndarray = field(default_factory=lambda: np.array([]))
    dt: np.ndarray = field(default_factory=lambda: np.array([]))
    D: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def plot_transient(self, ax: List[plt.Axes] = None, label: str = None, figsize=(8,3), include_D: bool = False) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        ax[0].plot(self.time, self.v, label=label)
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.0f}"))
        
        ax[0].set_xlabel('Time (us)')
        ax[0].set_ylabel('Voltage (V)')
        ax[0].set_title('Voltage Transient')
        
        ax[1].plot(self.time, self.i, label=label)        
        ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.0f}"))
        ax[1].set_xlabel('Time (us)')
        ax[1].set_ylabel('Current (A)')
        ax[1].set_title('Current Transient')
        
        
        ax[2].plot(self.time, self.dt, label=label)
        ax[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.0f}"))
        ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e6:.2f}"))
        ax[2].set_xlabel('Time (us)')
        ax[2].set_ylabel('Time Step (s)')
        ax[2].set_title('Time Step Transient')
      
        if include_D:
            # add a square wave plot for D
            D_square_wave_x, D_square_wave_y = self.to_square_wave(self.D)
            # transform the D_square_wave_x to match the time array
            D_square_wave_x = (D_square_wave_x - D_square_wave_x[0]) / (D_square_wave_x[-1] - D_square_wave_x[0]) * (self.time[-1] - self.time[0]) + self.time[0]
            
            # the square wave should be plotted with respect to a second y-axis on the right
            for axx in ax:
                axx.twinx().step(D_square_wave_x, D_square_wave_y, label='D', color='orange', where='post', linewidth=0.2, linestyle='--')
    
    @staticmethod
    def to_square_wave(arr):
        # Repeat each value, except the last, to create a step effect
        arr = np.asarray(arr)
        x = np.repeat(np.arange(len(arr)), 2)[1:]
        y = np.repeat(arr, 2)[:-1]
        return x, y

    @classmethod
    def from_h5(cls, h5_file: str, dataset_name: str, transient_number: int) -> 'TransientData':
        with h5py.File(h5_file, 'r') as f:
            data = f[dataset_name][f"subtransient_{transient_number}"]
            time = data['t'][:]
            v = data['v'][:]
            i = data['i'][:]
            dt = data['dt'][:]
            D = data["Dswitch"][:]
        return cls(time, v, i, dt, D)

# %% [markdown]
# These are signal sections taken from the overall MATLAB simulation results, using RK4:
# ![image.png](attachment:image.png)

# %%
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

tran_len = len(i1_n)

i_n = np.concatenate((i1_n, i2_n, i3_n), axis=0)
v_n = np.concatenate((v1_n, v2_n, v3_n), axis=0)
i_np1 = np.concatenate((i1_np1, i2_np1, i3_np1), axis=0)
v_np1 = np.concatenate((v1_np1, v2_np1, v3_np1), axis=0)
D = np.concatenate((D1, D2, D3), axis=0)
dt = np.concatenate((dt1, dt2, dt3), axis=0)

X = np.stack([i_n, v_n, D, dt], axis=1)
y = np.stack([i_np1, v_np1], axis=1)
Rload = np.concatenate((np.full_like(i1_n, Rload1), np.full_like(i2_n, Rload2), np.full_like(i3_n, Rload3)), axis=0)


# %%
class Normalizer:
    """Normalizer for the input data."""

    def __init__(self, X: np.ndarray):
        # Initialize the normalizer with the input data X
        if isinstance(X, np.ndarray):
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        elif isinstance(X, torch.Tensor):
            self.mean = X.mean(dim=0)
            self.std = X.std(dim=0)
        self.X = X

    def normalize(self) -> np.ndarray:
        # Normalize the input data X using the mean and std
        return (self.X - self.mean) / self.std

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        # Normalize the output data y using the mean and std
        return (y - self.mean[:2]) / self.std[:2]

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        # Denormalize the input data X_norm using the mean and std
        X_denorm = X_norm * self.std.reshape(1, -1) + self.mean.reshape(1, -1)
        return X_denorm

    def denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        # Denormalize the output data y_norm using the mean and std
        y_denorm = y_norm * self.std[:2].reshape(1, -1) + self.mean[:2].reshape(1, -1)
        return y_denorm
    
    def denormalize_intermediate(self, y_interm: np.ndarray) -> np.ndarray:
        # Denormalize the intermediate predictions
        y_interm_denorm = y_interm * self.std[:2].reshape(1, -1, 1) + self.mean[:2].reshape(1, -1, 1)
        return y_interm_denorm

# %%
def normalize_physical_params(
    L: torch.Tensor,
    RL: torch.Tensor,
    C: torch.Tensor,
    RC: torch.Tensor,
    Rdson: torch.Tensor,
    Rload1: torch.Tensor,
    Rload2: torch.Tensor,
    Rload3: torch.Tensor,
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
    Normalize the physical parameters to a logarithmic scale.
    The parameters are expected to be in their physical units.
    """
    L = torch.log(L * 1e4)  # L in H
    RL = torch.log(RL * 10)  # RL in Ohm
    C = torch.log(C * 1e4)  # C in F
    RC = torch.log(RC * 10)  # RC in Ohm
    Rdson = torch.log(Rdson * 10)  # Rdson in Ohm
    Rload1 = torch.log(Rload1)  # Rload in Ohm
    Rload2 = torch.log(Rload2)  # Rload in Ohm
    Rload3 = torch.log(Rload3)  # Rload in Ohm
    Vin = torch.log(Vin / 10)  # Vin in V
    Vf = torch.log(Vf)  # Vf is in V
    return L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, Vin, Vf


def denorm_physical_params(
    L: torch.Tensor,
    RL: torch.Tensor,
    C: torch.Tensor,
    RC: torch.Tensor,
    Rdson: torch.Tensor,
    Rload1: torch.Tensor,
    Rload2: torch.Tensor,
    Rload3: torch.Tensor,
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
    L = torch.exp(L) * 1e-4  
    RL = torch.exp(RL) * 1e-1
    C = torch.exp(C) * 1e-4  
    RC = torch.exp(RC) * 1e-1
    Rdson = torch.exp(Rdson) * 1e-1  
    Rload1 = torch.exp(Rload1) 
    Rload2 = torch.exp(Rload2)
    Rload3 = torch.exp(Rload3)
    Vin = torch.exp(Vin) * 10  
    Vf = torch.exp(Vf)  # Vf is in V
    return L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, Vin, Vf


# %%

class PINNBuck(nn.Module):
    """Physics‑Informed Neural Network for buck converter parameter estimation."""

    def __init__(
        self,
        layers: List[int],
        steps_irk: int,
        normalizer: Normalizer,
        tran_length: int = 238,
    ) -> None:
        super().__init__()

        self.normalizer = normalizer
        self.steps_irk = max(steps_irk, 1)
        self.tran_length = tran_length

        # ------------------------------------------------------------------
        # Trainable log‑parameters (initialised to TF defaults)
        # ------------------------------------------------------------------
        to_torch_tensor = lambda x: torch.tensor([x], dtype=torch.float32)
        log_physical_params = normalize_physical_params(
            to_torch_tensor(L0),
            to_torch_tensor(RL0),
            to_torch_tensor(C0),
            to_torch_tensor(RC0),
            to_torch_tensor(Rdson0),
            to_torch_tensor(Rload10),
            to_torch_tensor(Rload20),
            to_torch_tensor(Rload30),
            to_torch_tensor(Vin0),
            to_torch_tensor(VF),
        )
        
        self.log_L, self.log_RL, self.log_C, self.log_RC, \
        self.log_Rdson, self.log_Rload1, self.log_Rload2, self.log_Rload3,\
        self.log_vin, self.log_vF = log_physical_params
                    
        # ------------------------------------------------------------------
        # Neural network (fully connected, tanh activations)
        # ------------------------------------------------------------------
        layers_in = []
        for in_f, out_f in zip(layers[:-1], layers[1:]):
            layers_in.append(nn.Linear(in_f, out_f))
            if out_f != layers[-1]:  # last layer linear
                layers_in.append(nn.Tanh())
        self.fnn = nn.Sequential(*layers_in)

        # ------------------------------------------------------------------
        # IRK (Butcher tableau): pre‑computed constants as buffers
        # ------------------------------------------------------------------
        tmp = np.loadtxt(
            f"Butcher_tableau/Butcher_IRK{self.steps_irk}.txt", ndmin=2, dtype=np.float32
        )
        weights = tmp[: self.steps_irk * self.steps_irk + self.steps_irk].reshape(
            self.steps_irk + 1, self.steps_irk
        )
        irk_alpha = weights[:-1]
        irk_beta = weights[-1:]
        self.register_buffer("irk_alpha", torch.tensor(irk_alpha))
        self.register_buffer("irk_beta", torch.tensor(irk_beta))
        self.register_buffer(
            "irk_times", torch.tensor(tmp[self.steps_irk * self.steps_irk + self.steps_irk :])
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through FNN & split into current/voltage."""
        out = self.fnn(X)
        current_intermediate_predictions = out[:, : self.steps_irk]
        voltage_intermediate_predictions = out[:, self.steps_irk : 2 * self.steps_irk]
        return current_intermediate_predictions, voltage_intermediate_predictions


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _PI_head(
        self, i_interm: torch.Tensor, v_interm: torch.Tensor, 
        D: torch.Tensor, dt: torch.Tensor
        ):
        
        # get physical parameters
        L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF = denorm_physical_params(
            self.log_L, self.log_RL, self.log_C, self.log_RC,
            self.log_Rdson, self.log_Rload1, self.log_Rload2, self.log_Rload3,
            self.log_vin, self.log_vF
        )
        
        # denomralize the predictions
        y_interm_pred = self.normalizer.denormalize_intermediate(torch.stack([i_interm, v_interm], dim=1))
        i_iterm_pred = y_interm_pred[:, 0]
        v_iterm_pred = y_interm_pred[:, 1]
        
        
        
        rload = torch.cat(
            [
                torch.ones(self.tran_length, 1) * Rload1,
                torch.ones(self.tran_length, 1) * Rload2,
                torch.ones(self.tran_length, 1) * Rload3,
            ],
            dim=0,
        )
        
        def f(i: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            di = -((D * Rdson + RL) * i + v - D * vin + (1 - D) * vF) / L
            dv = (C * RC * rload * di + rload * i - v) / (C * (RC + rload))
            return di, dv

        i_deriv, v_deriv = f(i_iterm_pred, v_iterm_pred)

        # Backward prediction
        i_n = i_iterm_pred - dt * (i_deriv @ self.irk_alpha.T)
        v_n = v_iterm_pred - dt * (v_deriv @ self.irk_alpha.T)
        # Forward prediction
        i_np1 = i_iterm_pred + dt * (i_deriv @ (self.irk_beta - self.irk_alpha).T)
        v_np1 = v_iterm_pred + dt * (v_deriv @ (self.irk_beta - self.irk_alpha).T)

        return (i_n, v_n, i_np1, v_np1)

def compute_loss(
    preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y_n: torch.Tensor,
    y_np1: torch.Tensor,
) -> torch.Tensor:
    i_n_pred, v_n_pred, i_np1_pred, v_np1_pred = preds
    i_n, v_n = y_n[:, :1], y_n[:, 1:]
    i_np1, v_np1 = y_np1[:, :1], y_np1[:, 1:]
    
    loss = (
        torch.sum((i_n - i_n_pred) ** 2)
        + torch.sum((v_n - v_n_pred) ** 2)
        + torch.sum((i_np1 - i_np1_pred) ** 2)
        + torch.sum((v_np1 - v_np1_pred) ** 2)
    )
    return loss

# %%
adam_epochs: int = 45_000
adam_lr: float = 1e-3
lbfgs_epochs: int = 50_000


q = 20
layers = [4, 50, 50, 50, 50, 50, q * 2]


# transform all to tensors and move to device
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
normalizer_t = Normalizer(X_t)
X_norm_t = normalizer_t.normalize()


model = PINNBuck(
    layers=layers,  # Input: i_n, v_n, D, dt; Output: i_np1, v_np1
    steps_irk=20,
    normalizer=normalizer_t,
    tran_length=tran_len,
)

adam = torch.optim.Adam(
    model.parameters(),
    lr = adam_lr,
)

for epoch in range(adam_epochs):
    adam.zero_grad()
    i_interm, v_interm = model(X_norm_t)
    preds = model._PI_head(i_interm, v_interm, torch.tensor([D], dtype=torch.float32).T, torch.tensor([dt], dtype=torch.float32).T)

    xn = X_t[:, :2]  # i_n, v_n
    loss = compute_loss(preds, xn, y_t)

    loss.backward()
    adam.step()

    if epoch % 1000 == 0:
        L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vin, vF = model._phys_params()
        print(
            f"[Adam] iter={it:7d}, loss={loss.item():.3e}, "
            f"L={L.item():.3e}, RL={RL.item():.3e}, C={C.item():.3e}, "
            f"RC={RC.item():.3e}, Rdson={Rdson.item():.3e}, "
            f"Rload1={Rload1.item():.3e}, Rload2={Rload2.item():.3e}, "
            f"Rload3={Rload3.item():.3e}, Vin={vin.item():.3e}, VF={vF.item():.3e}"
        )


# %%
lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_epochs, tolerance_grad=1e-9)


def closure():
    lbfgs.zero_grad()
    l = compute_loss(model(X_norm_t), X[:, :2], y)
    l.backward()
    return l


print("Starting L‑BFGS optimisation … (this may take a while)")
lbfgs.step(closure)
print("L‑BFGS finished.")

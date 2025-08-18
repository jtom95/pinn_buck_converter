import torch
from pathlib import Path
import sys

t = torch.arange(0, 1000, 1, dtype=torch.float64)
y0 = torch.stack([torch.sin(t * 0.01), torch.cos(t * 0.01)], dim=-1)

y1 = y0 + torch.randn_like(y0) * 0.1
y2 = y0 + torch.randn_like(y0) * 0.1
y3 = y0 + torch.randn_like(y0) * 0.1

y = torch.stack([y1, y2, y3], dim=1)



sys.path.append(str(Path.cwd()))   

from pinn_buck.model.residual_time_correlation import ResidualDiagnosticsGaussian

gauss_residual_diagnostics = ResidualDiagnosticsGaussian(residuals=y)
vif = gauss_residual_diagnostics.quadloss_vif_from_residuals()

print("Variance Inflation Factor (VIF):", vif)
import torch
from typing import Callable, Dict, List, Optional
import matplotlib.pyplot as plt


class ResidualDiagnosticsGaussian:
    """
    Residual ACF, effective sample size heuristics, and visualization.

    Shapes:
        residuals: (N, T, 2)  # N=time within a transient, T=transients, 2=channels (i, v)
        acf:       (K, T, 2)  # K = nlags + 1, lag axis first
    """

    def __init__(self, residuals: torch.Tensor, nlags: Optional[int] = None):
        self.nlags = nlags  # default to N-1 if None at call time
        self.residuals = residuals
        assert (
            self.residuals.ndim == 3 and self.residuals.size(-1) == 2
        ), "Expected residuals (N, T, 2)"

    @torch.no_grad()
    def acf(self) -> torch.Tensor:
        """
        Per-transient, per-channel ACF along N (time) axis.

        Returns:
            acf: (K, T, 2), acf[0, t, c] == 1 if var>0 else 0
        """

        N, T, C = self.residuals.shape

        K = (self.nlags if self.nlags is not None else N - 1) + 1
        K = max(1, min(K, N))  # clamp to [1, N]

        # center per transient/channel
        x = self.residuals - self.residuals.mean(dim=0, keepdim=True)  # (N, T, 2)

        # denom: sum of squares per (T,2)
        denom = (x * x).sum(dim=0)  # (T, 2)

        acf = torch.zeros((K, T, C), dtype=x.dtype, device=x.device)

        # lag 0
        acf[0] = 0.0
        nonzero = denom > 0
        acf[0][nonzero] = 1.0  # rho(0)=1 if variance>0

        safe_denom = torch.where(nonzero, denom, torch.ones_like(denom))
        for k in range(1, K):
            num = (x[k:] * x[: N - k]).sum(dim=0)  # (T, 2)
            acf[k] = num / safe_denom

        # zero out where denom==0 (constant series)
        acf[:, ~nonzero] = 0.0
        acf[0, ~nonzero] = 0.0

        return acf  # (K, T, 2)

    @torch.no_grad()
    def per_channel_vif(self, max_lag: int) -> torch.Tensor:
        """
        Inflation factor for SSE-type losses (sum of squares) using squared ACF:
            infl = 1 + 2 * sum_{k=1..max_lag} rho_k^2

        Args:
            acf:    (K, T, 2)
            max_lag: use lags 1..max_lag

        Returns:
            infl: (T, 2) inflation per transient/channel (>=1)
        """
        acf = self.acf()
        K, T, C = acf.shape
        Kmax = max(1, min(max_lag, K - 1))
        s2 = (acf[1 : Kmax + 1] ** 2).sum(dim=0)  # (T, 2)
        return 1.0 + 2.0 * s2

    @torch.no_grad()
    def quadloss_vif_from_residuals(self, L_r: torch.Tensor, max_lag: int) -> torch.Tensor:
        """
        Variance inflation factor (VIF) for the quadratic loss sum_n r_n^T Sigma_r^{-1} r_n.

        Steps:
            1) Whiten instantaneous correlation via Cholesky of Sigma_r: L_r
            2) Compute lag-k cross-correlation matrix P_k of whitened y_n in each transient
            3) VIF = 1 + (2/d) * sum_k ||P_k||_F^2,   d=2

        Args:
            residuals: (N, T, 2)
            Sigma_r:   (2, 2) positive definite
            max_lag:   number of lags to include (<= N-1)

        Returns:
            vif: (T,) variance inflation factor per transient (>=1)
        """
        N, T, d = self.residuals.shape
        max_lag = max(1, min(max_lag, N - 1))

        # Cholesky and whitening
        rT = self.residuals.permute(1, 2, 0)  # (T, 2, N)
        yT = torch.linalg.solve_triangular(L_r, rT, upper=False)  # (T, 2, N)
        y = yT.permute(2, 0, 1)  # (N, T, 2)

        # center
        y = y - y.mean(dim=0, keepdim=True)

        # normalize to unit variance per channel per transient
        std = y.std(dim=0, unbiased=False, keepdim=True)  # (1, T, d)
        y_norm = y / std.clamp_min(1e-12)

        s = torch.zeros((T,), dtype=self.residuals.dtype, device=self.residuals.device)
        for k in range(1, max_lag + 1):
            Y1 = y_norm[k:]
            Y0 = y_norm[:-k]
            Pk = torch.einsum("ntc,ntd->tcd", Y1, Y0) / (N - k)  # correlation matrix
            nk = torch.linalg.norm(Pk, ord="fro", dim=(1, 2)) ** 2
            s = s + nk

        vif = 1.0 + (2.0 / d) * s
        return vif  # (T,)

    @torch.no_grad()
    def plot_acf(
        self,
        acf: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        residuals: torch.Tensor,
        group_label: Optional[str] = None,
        nlags: Optional[int] = None,
    ) -> List[plt.Figure]:
        """
        For each transient, show targets vs predictions, residuals, and ACF.
        """
        assert targets.shape == predictions.shape == residuals.shape, "Expect (N, T, 2)"
        N, T, C = targets.shape
        K = acf.size(0)
        show_K = min(K, nlags if nlags is not None else K)

        def to_np(t: torch.Tensor) -> torch.Tensor:
            return t.detach().cpu().numpy()

        acf_view = acf[:show_K]
        tgt = targets[:show_K] if show_K <= N else targets
        pred = predictions[:show_K] if show_K <= N else predictions
        resid = residuals[:show_K] if show_K <= N else residuals

        figs: List[plt.Figure] = []
        for t in range(T):
            fig, ax = plt.subplots(3, 2, constrained_layout=True, figsize=(10, 5))
            figs.append(fig)

            # 1) targets vs predictions
            ax[0, 0].plot(to_np(tgt[:, t, 0]), label="Target Current")
            ax[0, 1].plot(to_np(tgt[:, t, 1]), label="Target Voltage")
            ax[0, 0].plot(to_np(pred[:, t, 0]), "--", label="Predicted Current")
            ax[0, 1].plot(to_np(pred[:, t, 1]), "--", label="Predicted Voltage")
            ax[0, 0].set_title("Targets / Predictions — Current")
            ax[0, 1].set_title("Targets / Predictions — Voltage")
            for axx in ax[0]:
                axx.set_xlabel("Time")
                axx.legend()

            # 2) residuals
            ax[1, 0].plot(to_np(resid[:, t, 0]), label="Current")
            ax[1, 1].plot(to_np(resid[:, t, 1]), label="Voltage")
            ax[1, 0].set_title("Residuals — Current")
            ax[1, 1].set_title("Residuals — Voltage")
            for axx in ax[1]:
                axx.set_xlabel("Time")
                axx.set_ylabel("Residuals")
                axx.legend()

            # 3) ACF
            ax[2, 0].plot(to_np(acf_view[:, t, 0]), label="Current")
            ax[2, 1].plot(to_np(acf_view[:, t, 1]), label="Voltage")
            ax[2, 0].set_title("ACF — Current")
            ax[2, 1].set_title("ACF — Voltage")
            for axx in ax[2]:
                axx.set_xlabel("Lag")
                axx.set_ylabel("ACF")
                axx.legend()

            title = f"Transient {t+1}"
            if group_label:
                title += f" — {group_label}"
            fig.suptitle(title, fontsize=14)

        return figs

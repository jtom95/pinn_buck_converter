import torch
from typing import Callable, Dict, List, Optional, Literal
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
            acf[k] = (N / (N - k)) * num / safe_denom

        # zero out where denom==0 (constant series)
        acf[:, ~nonzero] = 0.0
        acf[0, ~nonzero] = 0.0

        return acf  # (K, T, 2)

    @torch.no_grad()
    def per_channel_vif(self, max_lag: int | None = None, bandwidth_rule: str = "n13") -> torch.Tensor:
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
        N = self.residuals.shape[0]

        if max_lag is None:
            if bandwidth_rule == "n13":
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            else:
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            M = min(M, N - 1)
        else:
            M = max(1, min(max_lag, N - 1))

        w = torch.linspace(1, 0, steps=M + 1, device=acf.device)[1:]  # Bartlett 1..M
        s2 = ((acf[1 : M + 1] ** 2) * w.view(-1, 1, 1)).sum(dim=0)
        return 1.0 + 2.0 * s2


    @torch.no_grad()
    def quadloss_vif_from_residuals(
        self,
        max_lag: int | None = None,
        shrinkage_alpha: float = 0.2,  # used by renorm_mode="matrix"
        jitter: float = 1e-8,
        unbiased: bool = False,
        bartlett: bool = True,
        bandwidth_rule: str = "n13",  # "n13" => M = floor(1.5 * N^(1/3))
        renorm_mode: str = "corr_shrink",  # "none" | "std" | "matrix" | "corr_shrink"
        beta_corr: float = 0.1,  # used by "corr_shrink" (light whitening of correlation)
    ) -> torch.Tensor:
        """
        VIF for quadratic loss using *empirical* instantaneous covariance per transient.

        Steps per transient:
        1) Center residuals over time -> y.
        2) Optional lag-0 renormalization (renorm_mode):
            - "none":         use y as-is.
            - "std":          per-channel z-score.
            - "matrix":       whiten by chol( (1-a)I + a*S ), S = cov(y).
            - "corr_shrink":  z-score, then *lightly* whiten by chol( (1-b)I + b*R ),
                            R = correlation of z-scored y. (Default: beta_corr=0.1)
        3) Compute lag-k cross-covariances P_k of renormalized series.
        4) VIF = 1 + (2/d) * sum_k w_k * ||P_k||_F^2
        """
        N, T, d = self.residuals.shape
        assert d == 2, "expected 2 channels"

        # --- bandwidth ---
        if max_lag is None:
            if bandwidth_rule == "n13":
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            else:
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            M = min(M, N - 1)
        else:
            M = max(1, min(max_lag, N - 1))

        # --- center over time ---
        y = self.residuals - self.residuals.mean(dim=0, keepdim=True)  # (N,T,2)

        # --- lag-0 renormalization (choose one) ---
        if renorm_mode == "none":
            y_w = y

        elif renorm_mode == "std":
            std = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)  # (1,T,2)
            y_w = y / std

        elif renorm_mode == "matrix":
            # empirical instantaneous covariance per transient
            S = torch.einsum("ntc,ntd->tcd", y, y) / N  # (T,2,2)
            I = torch.eye(d, dtype=y.dtype, device=y.device).expand_as(S)
            S_sh = (1 - shrinkage_alpha) * I + shrinkage_alpha * S
            S_sh = S_sh + jitter * I
            L = torch.linalg.cholesky(S_sh)  # (T,2,2)
            yT = y.permute(1, 2, 0)  # (T,2,N)
            y_wT = torch.linalg.solve_triangular(L, yT, upper=False)  # (T,2,N)
            y_w = y_wT.permute(2, 0, 1)  # (N,T,2)

        elif renorm_mode == "corr_shrink":
            # 1) per-channel z-score (diagonal fix)
            std = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
            yz = y / std  # (N,T,2)

            # 2) correlation per transient and gentle whitening
            R = torch.einsum("ntc,ntd->tcd", yz, yz) / N  # (T,2,2), diag ~ I
            I = torch.eye(d, dtype=y.dtype, device=y.device).expand_as(R)
            R_sh = (1 - beta_corr) * I + beta_corr * R
            R_sh = R_sh + jitter * I
            L = torch.linalg.cholesky(R_sh)  # (T,2,2)
            yzT = yz.permute(1, 2, 0)  # (T,2,N)
            y_wT = torch.linalg.solve_triangular(L, yzT, upper=False)  # (T,2,N)
            y_w = y_wT.permute(2, 0, 1)  # (N,T,2)

        else:
            raise ValueError(f"Unknown renorm_mode: {renorm_mode}")

        # --- weights (Bartlett or flat) ---
        w = torch.ones((M,), dtype=y.dtype, device=y.device)
        if bartlett:
            w = 1.0 - torch.arange(1, M + 1, device=y.device, dtype=y.dtype) / (M + 1)

        # --- accumulate ||P_k||_F^2 ---
        s = torch.zeros((T,), dtype=y.dtype, device=y.device)
        for k in range(1, M + 1):
            Pk = torch.einsum("ntc,ntd->tcd", y_w[k:], y_w[:-k]) / (N - k)  # (T,2,2)
            if unbiased:
                Pk = (N / (N - k)) * Pk
            s = s + w[k - 1] * (torch.linalg.norm(Pk, ord="fro", dim=(1, 2)) ** 2)

        vif = 1.0 + (2.0 / d) * s
        return vif

    @torch.no_grad()
    def quadloss_vif_from_residuals1(
        self,
        max_lag: int | None = None,
        shrinkage_alpha: float = 0.2,
        jitter: float = 1e-8,
        unbiased: bool = False,
        bartlett: bool = True,
        bandwidth_rule: str = "n13",  # "n13" => M = floor(1.5*N^(1/3))
    ) -> torch.Tensor:
        """
        VIF for quadratic loss using *empirical* instantaneous covariance per transient.

        Steps per transient:
        1) Center residuals over time.
        2) Compute empirical lag-0 covariance S_t.
        3) Shrink S_t toward I:  S_sh = (1-a) I + a S_t.
        4) Whiten by chol(S_sh)^{-1}.
        5) Compute lag-k cross-covariances P_k of whitened series.
        6) VIF = 1 + (2/d) * sum_k w_k * ||P_k||_F^2

        Args:
            max_lag: if None, use bandwidth_rule (default M = floor(1.5 * N^(1/3))).
            shrinkage_alpha: shrinkage toward I in [0,1].
            jitter: tiny diagonal added before Cholesky.
            unbiased: apply N/(N-k) factor (optional).
            bartlett: use Bartlett taper.
            bandwidth_rule: currently "n13" only.

        Returns:
            vif: (T,)
        """
        N, T, d = self.residuals.shape
        assert d == 2, "expected 2 channels"
        if max_lag is None:
            if bandwidth_rule == "n13":
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            else:
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            M = min(M, N - 1)
        else:
            M = max(1, min(max_lag, N - 1))

        # center over time
        y = self.residuals - self.residuals.mean(dim=0, keepdim=True)  # (N,T,2)

        # empirical instantaneous covariance per transient
        S = torch.einsum("ntc,ntd->tcd", y, y) / N  # (T,2,2)
        I = torch.eye(d, dtype=y.dtype, device=y.device).expand_as(S)
        S_sh = (1 - shrinkage_alpha) * I + shrinkage_alpha * S
        S_sh = S_sh + jitter * I

        # whiten by chol(S_sh)^{-1}
        L = torch.linalg.cholesky(S_sh)  # (T,2,2)
        yT = y.permute(1, 2, 0)  # (T,2,N)
        y_wT = torch.linalg.solve_triangular(L, yT, upper=False)  # (T,2,N)
        y_w = y_wT.permute(2, 0, 1)  # (N,T,2)

        # Bartlett weights (or flat)
        w = torch.ones((M,), dtype=y.dtype, device=y.device)
        if bartlett:
            w = 1.0 - torch.arange(1, M + 1, device=y.device, dtype=y.dtype) / (M + 1)

        # accumulate ||P_k||_F^2
        s = torch.zeros((T,), dtype=y.dtype, device=y.device)
        for k in range(1, M + 1):
            Pk = torch.einsum("ntc,ntd->tcd", y_w[k:], y_w[:-k]) / (N - k)  # (T,2,2)
            if unbiased:
                Pk = (N / (N - k)) * Pk
            s = s + w[k - 1] * (torch.linalg.norm(Pk, ord="fro", dim=(1, 2)) ** 2)

        vif = 1.0 + (2.0 / d) * s
        return vif
    
    @torch.no_grad()
    def quadloss_vif_from_residuals_theoretical_Sigma(
        self,
        L_r: torch.Tensor,
        max_lag: int | None = None,
        unbiased: bool = False,
        bartlett: bool = True,
        bandwidth_rule: str = "n13",   # "n13" => M = floor(1.5*N^(1/3))
        renorm_mode: Literal["none", "std", "corr_shrink"] = "corr_shrink",
        beta_corr: float = 0.1,        # shrinkage factor for "corr_shrink"
        jitter: float = 1e-8,
    ) -> torch.Tensor:
        """
        Variance inflation factor (VIF) for the quadratic loss
            sum_n r_n^T Sigma_r^{-1} r_n
        using *theoretical* Sigma_r for initial whitening.

        Steps:
        1) Whiten instantaneous correlation via Cholesky of Sigma_r: L_r
        2) Center over time
        3) Optional lag-0 renormalization (renorm_mode):
            - "none":         keep y as-is
            - "std":          per-channel z-score
            - "corr_shrink":  z-score, then gently whiten by correlation shrinkage
        4) Compute lag-k cross-covariance matrices P_k
        5) VIF = 1 + (2/d) * sum_k w_k * ||P_k||_F^2

        Args:
            L_r: (2,2) or (T,2,2) Cholesky of Sigma_r (lower-triangular)
            max_lag: number of lags to include (if None, use bandwidth_rule)
            unbiased: apply N/(N-k) correction factor
            bartlett: apply Bartlett taper (default True)
            bandwidth_rule: "n13" => floor(1.5 * N^(1/3)) lags
            renorm_mode: "none" | "std" | "corr_shrink"
            beta_corr: shrinkage parameter for "corr_shrink"
            jitter: small diagonal for Cholesky stability

        Returns:
            vif: (T,) variance inflation factor per transient (>=1)
        """
        N, T, d = self.residuals.shape
        assert d == 2, "expected 2 channels"

        # --- bandwidth ---
        if max_lag is None:
            if bandwidth_rule == "n13":
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            else:
                M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            M = min(M, N - 1)
        else:
            M = max(1, min(max_lag, N - 1))

        # --- theoretical whitening ---
        rT = self.residuals.permute(1, 2, 0)                     # (T,d,N)
        yT = torch.linalg.solve_triangular(L_r, rT, upper=False) # (T,d,N)
        y  = yT.permute(2, 0, 1)                                 # (N,T,d)

        # --- center in time ---
        y = y - y.mean(dim=0, keepdim=True)

        # --- optional renormalization ---
        if renorm_mode == "none":
            y_w = y

        elif renorm_mode == "std":
            std = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
            y_w = y / std

        elif renorm_mode == "corr_shrink":
            # 1) z-score
            std = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
            yz = y / std
            # 2) empirical correlation matrices
            R = torch.einsum("ntc,ntd->tcd", yz, yz) / N          # (T,2,2)
            I = torch.eye(d, dtype=y.dtype, device=y.device).expand_as(R)
            R_sh = (1 - beta_corr) * I + beta_corr * R
            R_sh = R_sh + jitter * I
            Lcorr = torch.linalg.cholesky(R_sh)                   # (T,2,2)
            yzT   = yz.permute(1, 2, 0)                           # (T,2,N)
            yz_wT = torch.linalg.solve_triangular(Lcorr, yzT, upper=False)
            y_w   = yz_wT.permute(2, 0, 1)                        # (N,T,2)

        else:
            raise ValueError(f"Unknown renorm_mode: {renorm_mode}")

        # --- weights ---
        w = torch.ones((M,), dtype=y.dtype, device=y.device)
        if bartlett:
            w = 1.0 - torch.arange(1, M + 1, device=y.device, dtype=y.dtype) / (M + 1)

        # --- accumulate ||P_k||_F^2 ---
        s = torch.zeros((T,), dtype=y.dtype, device=y.device)
        for k in range(1, M + 1):
            Pk = torch.einsum("ntc,ntd->tcd", y_w[k:], y_w[:-k]) / (N - k)  # (T,2,2)
            if unbiased:
                Pk = (N / (N - k)) * Pk
            s = s + w[k - 1] * (torch.linalg.norm(Pk, ord="fro", dim=(1, 2)) ** 2)

        vif = 1.0 + (2.0 / d) * s
        return vif

    @torch.no_grad()
    def quadloss_vif_from_residuals_theoretical_Sigma1(
        self, L_r: torch.Tensor, 
        max_lag: int | None = None, 
        use_std_renorm: bool = True,
        beta_corr: float = 0.1,
        unbiased: bool = False
        ) -> torch.Tensor:
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
        if max_lag is None:
            M = max(1, int(1.5 * (N ** (1.0 / 3.0))))
            M = min(M, N - 1)
        else:
            M = max(1, min(max_lag, N - 1))

        rT = self.residuals.permute(1, 2, 0)                     # (T, d, N)
        yT = torch.linalg.solve_triangular(L_r, rT, upper=False) # (T, d, N)
        y  = yT.permute(2, 0, 1)                                 # (N, T, d)

        # center over time index N
        y = y - y.mean(dim=0, keepdim=True)

        if use_std_renorm:
            std = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
            y = y / std

            # 2) light correlation whitening with shrinkage (beta small)
            # correlation per transient after std renorm
            R = torch.einsum("ntc,ntd->tcd", y, y) / y.shape[0]          # (T,2,2), diag≈I
            I = torch.eye(2, dtype=y.dtype, device=y.device).expand_as(R)
            R_sh = (1 - beta_corr) * I + beta_corr * R
            L = torch.linalg.cholesky(R_sh + 1e-8 * I)                    # (T,2,2)

            # apply light whitening
            yT  = y.permute(1,2,0)                                        # (T,2,N)
            y2T = torch.linalg.solve_triangular(L, yT, upper=False)       # (T,2,N)
            y   = y2T.permute(2,0,1)    

        s = torch.zeros((T,), dtype=y.dtype, device=y.device)
        for k in range(1, M + 1):
            w = 1.0 - k / (M + 1)                          # Bartlett
            Pk = torch.einsum("ntc,ntd->tcd", y[k:], y[:-k]) / (N - k)
            if unbiased:
                Pk = (N / (N - k)) * Pk
            s = s + w * (torch.linalg.norm(Pk, ord="fro", dim=(1, 2)) ** 2)

        vif = 1.0 + (2.0 / d) * s
        return vif

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

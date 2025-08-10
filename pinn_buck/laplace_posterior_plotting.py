import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Iterable, Optional, Literal
from scipy.stats import lognorm, norm
from pinn_buck.constants import ParameterConstants

from typing import Protocol, runtime_checkable

from .laplace_posterior_fitting import LaplacePosterior


@runtime_checkable
class RVLike(Protocol):
    def pdf(self, x: float | np.ndarray) -> np.ndarray: ...
    def ppf(self, q: float | np.ndarray) -> np.ndarray: ...
    def mean(self) -> float: ...
    def std(self) -> float: ...


class LaplacePosteriorPlotter:
    """
    Plotter for LaplacePosterior objects.
    Relies on LaplacePosterior.gaussian_approx and .lognormal_approx
    (already in physical units), never on raw theta_log/Sigma_log.
    """

    NOMINAL = ParameterConstants.NOMINAL
    REL_TOL = ParameterConstants.REL_TOL
    _SCALE = ParameterConstants.SCALE
    TRUE_PARAMS = ParameterConstants.TRUE

    # ---------- small helpers ----------

    @classmethod
    def _prior_dist_for(cls, name: str) -> RVLike:
        nominal = cls.NOMINAL.get_from_iterator_name(name)
        rel_tol = cls.REL_TOL.get_from_iterator_name(name)
        sigma_prior = np.log(1.0 + rel_tol)
        mu_prior = np.log(nominal)
        return lognorm(s=sigma_prior, scale=np.exp(mu_prior))

    @staticmethod
    def _format_axis(ax: plt.Axes, name: str) -> None:
        # add simple unit formatting if desired
        if name == "L":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
            ax.set_xlabel("[mH]")
        elif name == "C":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
            ax.set_xlabel("[μF]")

    # ---------- public API ----------

    def plot_laplace_posteriors(
        cls,
        lfit: "LaplacePosterior",
        include_true_params: bool = True,
        ncols: int = 2,
        fig_size=(12, 14),
        show: bool = True,
    ):
        """
        For a single LaplacePosterior, plot prior (log-normal) and Laplace posteriors
        (Gaussian + LogNormal) for each parameter.
        """
        param_names = list(lfit.param_names)
        nrows = int(np.ceil(len(param_names) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
        axes = np.atleast_1d(axes).ravel()

        gauss_dists = lfit.gaussian_approx
        logn_dists = lfit.lognormal_approx

        for i, name in enumerate(param_names):
            ax = axes[i]
            ax.set_title(name)
            ax.set_yticks([])

            # Prior
            prior = cls._prior_dist_for(name)
            x_prior = np.linspace(prior.ppf(0.001), prior.ppf(0.999), 500)
            y_prior = prior.pdf(x_prior)
            ax.plot(x_prior, y_prior, label="Prior (log-normal)", color="blue", linewidth=1)

            # Posterior (Gaussian, physical units)
            g = gauss_dists[i]
            g_mean, g_std = g.mean(), g.std()
            xg = np.linspace(g_mean - 4 * g_std, g_mean + 4 * g_std, 500)
            yg = g.pdf(xg)
            ax.plot(xg, yg, label="Laplace Posterior (Gaussian)", color="orange", linewidth=1)

            # Posterior (LogNormal, physical units)
            ln = logn_dists[i]
            xl = np.linspace(ln.ppf(0.001), ln.ppf(0.999), 500)
            yl = ln.pdf(xl)
            ax.plot(xl, yl, label="Laplace Posterior (log-normal)", color="black", linewidth=2)

            # Markers
            mu_post = float(lfit.theta_phys[i].cpu().numpy())  # MAP in physical units
            ax.axvline(mu_post, color="purple", linestyle="-.", label="MAP Estimate", linewidth=1)
            if include_true_params:
                true_val = cls.TRUE_PARAMS.get_from_iterator_name(name)
                ax.axvline(true_val, color="red", linestyle="--", label="TRUE", linewidth=1)

            # optional: other run estimates (if you want to add later)
            # if runs_ordered:
            #     for lbl, run in runs_ordered.items():
            #         est = getattr(run.best_parameters, name)
            #         ax.axvline(est, linestyle=":", label=f"{lbl} estimate", linewidth=1)

            cls._format_axis(ax, name)
            ax.legend(fontsize="x-small", loc="upper right")

        # hide any extra axes
        for j in range(len(param_names), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Prior (log-normal) and Laplace Posterior for Each Parameter", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        if show:
            plt.show()
        return fig, axes

    def plot_single_laplace_posterior(
        cls,
        param_name: str,
        lfit: "LaplacePosterior",
        ax: plt.Axes,
        label: str,
        style: str = "log-normal",  # or "gaussian"
        color: Optional[str] = None,
        plot_prior: bool = True,
        plot_true_params: bool = True,
        show_map_marker: bool = True,
        marker_kwargs: Optional[dict] = None,
    ):
        """
        Plot one posterior PDF (Gaussian or LogNormal) for a single parameter.
        Uses LaplacePosterior's distributions in physical units.
        """
        try:
            idx = lfit.param_names.index(param_name)
        except ValueError:
            raise ValueError(
                f"Parameter '{param_name}' not found in lfit.param_names={lfit.param_names}"
            )

        ax.set_title(param_name)
        ax.set_yticks([])

        # Optional prior
        if plot_prior:
            prior = cls._prior_dist_for(param_name)
            xp = np.linspace(prior.ppf(0.001), prior.ppf(0.999), 500)
            yp = prior.pdf(xp)
            ax.plot(xp, yp, label="Prior (log-normal)", color="black", linewidth=2)

        # Choose posterior style
        if style == "gaussian":
            dist = lfit.gaussian_approx[idx]
            mean, std = dist.mean(), dist.std()
            x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
        elif style == "log-normal":
            dist = lfit.lognormal_approx[idx]
            x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
        else:
            raise ValueError("style must be 'gaussian' or 'log-normal'.")

        (line,) = ax.plot(x, dist.pdf(x), label=label, color=color, linewidth=1)
        line_color = line.get_color() if color is None else color

        # MAP marker (at physical MAP value)
        mu_post = float(lfit.theta_phys[idx].cpu().numpy())
        if show_map_marker:
            y_map = dist.pdf(mu_post)
            mk = {"marker": "s", "color": line_color, "s": 30, "zorder": 5}
            if marker_kwargs:
                mk.update(marker_kwargs)
            ax.scatter([mu_post], [y_map], **mk)

        if plot_true_params:
            true_val = cls.TRUE_PARAMS.get_from_iterator_name(param_name)
            ax.axvline(true_val, color="red", linestyle="--", label="TRUE", linewidth=1)

        cls._format_axis(ax, param_name)
        return ax

    @classmethod
    def plot_all_laplace_posteriors_grid(
        cls,
        lfits: Dict[str, "LaplacePosterior"],  # label -> posterior
        plot_true_params: bool = True,
        skip_labels: Iterable[str] = ("ideal",),
        style: Literal["log-normal", "gaussian"] = "log-normal",
        ncols: int = 2,
        fig_size=(12, 14),
        show: bool = True,
    ):
        """
        Plot PDFs for all parameters in a grid, overlaying multiple Laplace posteriors.
        Prior and true-value marker are drawn once per subplot.
        """
        # Use the first posterior as reference for parameter names/order
        if not lfits:
            raise ValueError("lfits is empty.")
        first = next(iter(lfits.values()))
        param_names = list(first.param_names)

        nrows = int(np.ceil(len(param_names) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
        axes = np.atleast_1d(axes).ravel()

        for i, name in enumerate(param_names):
            ax = axes[i]
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_title(name)
            ax.set_yticks([])

            # Prior once
            prior = cls._prior_dist_for(name)
            xp = np.linspace(prior.ppf(0.001), prior.ppf(0.999), 500)
            yp = prior.pdf(xp)
            ax.plot(xp, yp, label="Prior (log-normal)", color="black", linewidth=2)

            # TRUE once
            if plot_true_params:
                true_val = cls.TRUE_PARAMS.get_from_iterator_name(name)
                ax.axvline(true_val, color="red", linestyle="--", label="TRUE", linewidth=1)

            # Plot each posterior
            for lbl, lfit in lfits.items():
                if lbl in set(skip_labels):
                    continue
                cls.plot_single_laplace_posterior(
                    cls=cls,
                    param_name=name,
                    lfit=lfit,
                    ax=ax,
                    label=lbl,
                    style=style,
                    color=None,
                    plot_prior=False,  # already drawn
                    plot_true_params=False,
                    show_map_marker=True,
                    marker_kwargs={"label": None},  # avoid duplicate legend entries
                )

            cls._format_axis(ax, name)
            ax.legend(fontsize="x-small", loc="upper right")

        # hide extras
        for j in range(len(param_names), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Laplace Posteriors", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        if show:
            plt.show()
        return fig, axes

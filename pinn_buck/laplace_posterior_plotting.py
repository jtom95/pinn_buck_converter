import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Iterable, Optional, Literal, Tuple, Union
from scipy.stats import lognorm, norm
from pinn_buck.constants import ParameterConstants
import itertools

from typing import Protocol, runtime_checkable

from .laplace_posterior_fitting import LaplacePosterior
from .config import Parameters


AxesLike = Union[plt.Axes, np.ndarray]


def _coerce_axes_grid(ax: Optional[AxesLike], nrows: int, ncols: int, fig_size):
    """
    Return (fig, axes2d, axes_flat).
    If ax is None -> create fig,axes.
    If ax is a single Axes and nrows*ncols==1 -> wrap it.
    If ax is a 2D array of Axes with correct shape -> use it.
    """
    if ax is None:
        fig, axes2d = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    else:
        # ax can be a single Axes (only valid for 1x1) or a np.ndarray of Axes with shape (nrows, ncols)
        if isinstance(ax, plt.Axes):
            if nrows != 1 or ncols != 1:
                raise ValueError(f"Got a single Axes but need a grid {nrows}x{ncols}.")
            axes2d = np.array([[ax]])
            fig = ax.figure
        else:
            if not isinstance(ax, np.ndarray):
                raise TypeError(
                    f"`ax` must be a matplotlib Axes or ndarray of Axes, got {type(ax)}"
                )
            if ax.shape != (nrows, ncols):
                raise ValueError(f"Axes array has shape {ax.shape}, expected {(nrows, ncols)}")
            axes2d = ax
            # take the figure from the first Axes object
            fig = axes2d.flat[0].figure
    axes_flat = axes2d.ravel()
    return fig, axes2d, axes_flat


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

    # ---------- small helpers ----------

    @classmethod
    def _safe_color_generator(cls):
        """
        Yield colors from matplotlib's prop_cycle, skipping red.
        Wraps around indefinitely so it can be reused across subplots.
        """
        safe_colors = [
            c
            for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
            if c.lower() not in ("r", "#ff0000", "#d62728")
        ]
        return itertools.cycle(safe_colors)

    @classmethod
    def get_safecolor(cls):
        if not hasattr(cls, "_color_cycle"):
            cls._color_cycle = cls._safe_color_generator()
        return next(cls._color_cycle)

    @classmethod
    def _prior_dist_for(cls, mu, sigma) -> RVLike:
        return lognorm(s=sigma, scale=mu)

    @staticmethod
    def _format_axis(ax: plt.Axes, name: str) -> None:
        # add simple unit formatting if desired
        if name == "L":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.2f}"))
            ax.set_xlabel("[mH]")
        elif name == "C":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.2f}"))
            ax.set_xlabel("[μF]")

    @staticmethod
    def plot_prior_only(
        ax: plt.Axes,
        mu: float,
        sigma: float,
        interval: Tuple[float, float] = (0.01, 0.99),
        *,
        color: Optional[str] = "black",
        linestyle: str = "-",
        linewidth: float = 2.0,
        label: Optional[str] = "Prior (log-normal)",
        show_marker: bool = True,
        marker_kwargs: Optional[dict] = None,
    ) -> plt.Axes:
        """
        Draw only the lognormal prior PDF on `ax`, without plotting any posterior.
        - mu, sigma are the prior's parameters in physical space, consistent with your Parameters.
        - interval is the (lo, hi) quantile range for the x-limits.
        """
        prior = lognorm(s=sigma, scale=mu)
        lo, hi = interval
        x = np.linspace(prior.ppf(lo), prior.ppf(hi), 500)
        y = prior.pdf(x)
        (line,) = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

        if show_marker:
            y_mu = prior.pdf(mu)
            mk = {"marker": "s", "color": line.get_color(), "s": 30, "zorder": 5}
            if marker_kwargs:
                mk.update(marker_kwargs)
            ax.scatter([mu], [y_mu], **mk)

        return ax

    # ---------- public API ----------
    @classmethod
    def plot_laplace_posteriors(
        cls,
        lfit: "LaplacePosterior",
        true_params: Optional[Parameters] = None,
        prior_mu: Optional[Parameters] = None,
        prior_sigma: Optional[Parameters] = None,
        prior_pdf_interval: Optional[Tuple[float, float]] = None,
        ncols: int = 2,
        fig_size=(12, 14),
        ax: Optional[plt.Axes] = None,
        add_legend: bool = True
    ):
        """
        For a single LaplacePosterior, plot prior (log-normal) and Laplace posteriors
        (Gaussian + LogNormal) for each parameter.
        """

        param_names = list(lfit.param_names)
        nrows = int(np.ceil(len(param_names) / ncols))
        fig, axes2d, axes = _coerce_axes_grid(ax, nrows, ncols, fig_size)

        gauss_dists = lfit.gaussian_approx
        logn_dists = lfit.lognormal_approx

        for i, name in enumerate(param_names):
            ax = axes[i]
            ax.set_title(name)
            ax.set_yticks([])

            # Prior
            if prior_mu is not None and prior_sigma is not None:
                cls.plot_prior_only(
                    ax=ax,
                    mu=prior_mu.get_from_iterator_name(name),
                    sigma=prior_sigma.get_from_iterator_name(name),
                    interval=prior_pdf_interval,
                    color="blue",
                    label="Prior (log-normal)",
                    linewidth=1,
                )
                # q_lo, q_hi = prior_pdf_interval or (0.001, 0.999)
                # xp = np.linspace(prior.ppf(q_lo), prior.ppf(q_hi), 500)
                # # get the parameter corresponding to name
                # mu0 = prior_mu.get_from_iterator_name(name)
                # sigma0 = prior_sigma.get_from_iterator_name(name)
                # prior = cls._prior_dist_for(mu0, sigma0)
                # x_prior = np.linspace(prior.ppf(prior_pdf_interval[0]), prior.ppf(prior_pdf_interval[1]), 500)
                # y_prior = prior.pdf(x_prior)
                # ax.plot(x_prior, y_prior, label="Prior (log-normal)", color="blue", linewidth=1)

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
            if true_params is not None:
                true_val = true_params.get_from_iterator_name(name)
                ax.axvline(true_val, color="red", linestyle="--", label="TRUE", linewidth=1)

            # optional: other run estimates (if you want to add later)
            # if runs_ordered:
            #     for lbl, run in runs_ordered.items():
            #         est = getattr(run.best_parameters, name)
            #         ax.axvline(est, linestyle=":", label=f"{lbl} estimate", linewidth=1)

            cls._format_axis(ax, name)
            if add_legend:
                ax.legend(fontsize="x-small", loc="upper right")

        # hide any extra axes
        for j in range(len(param_names), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Prior (log-normal) and Laplace Posterior for Each Parameter", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        return fig, axes

    @classmethod
    def plot_single_laplace_posterior(
        cls,
        param_name: str,
        lfit: "LaplacePosterior",
        ax: plt.Axes,
        label: str,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        color: Optional[str] = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        prior_mu: Optional[float] = None,
        prior_sigma: Optional[float] = None,
        true_param: Optional[float] = None,
        show_map_marker: bool = True,
        marker_kwargs: Optional[dict] = None,
        pdf_interval: Optional[Tuple[float, float]] = None,
        prior_pdf_interval: Optional[Tuple[float, float]] = None,
        add_legend: bool = True,
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
        if prior_mu is not None and prior_sigma is not None:
            cls.plot_prior_only(
                ax=ax,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                interval=prior_pdf_interval,
                label="Prior (log-normal)",
                color="black",
                linewidth=2
            )
            # prior = cls._prior_dist_for(prior_mu, prior_sigma)
            # q_lo, q_hi = prior_pdf_interval or (0.01, 0.99)
            # xp = np.linspace(prior.ppf(q_lo), prior.ppf(q_hi), 500)
            # yp = prior.pdf(xp)
            # (line,) = ax.plot(xp, yp, label="Prior (log-normal)", color="black", linewidth=2)
            # line_color = line.get_color() if color is None else color
            # if show_map_marker:
            #     y_prior = prior.pdf(prior_mu)
            #     mk = {"marker": "s", "color": line_color, "s": 30, "zorder": 5}
            #     if marker_kwargs:
            #         mk.update(marker_kwargs)
            #     ax.scatter([prior_mu], [y_prior], **mk)

        # Choose posterior style
        if distribution_type == "gaussian":
            dist = lfit.gaussian_approx[idx]
            mean, std = dist.mean(), dist.std()
            x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
        elif distribution_type == "log-normal":
            dist = lfit.lognormal_approx[idx]
            q_lo, q_hi = pdf_interval or (0.01, 0.99)
            x = np.linspace(dist.ppf(q_lo), dist.ppf(q_hi), 500)
        else:
            raise ValueError("distribution_type must be 'gaussian' or 'log-normal'.")

        color = color or cls.get_safecolor()

        (line,) = ax.plot(
            x, dist.pdf(x), label=label, color=color, linewidth=linewidth, linestyle=linestyle
            )
        line_color = line.get_color() if color is None else color

        # MAP marker (at physical MAP value)
        mu_post = float(lfit.theta_phys[idx].cpu().numpy())
        if show_map_marker:
            y_map = dist.pdf(mu_post)
            mk = {"marker": "s", "color": line_color, "s": 30, "zorder": 5}
            if marker_kwargs:
                mk.update(marker_kwargs)
            ax.scatter([mu_post], [y_map], **mk)

        if true_param is not None:
            ax.axvline(true_param, color="red", linestyle="--", label="TRUE", linewidth=1)

        cls._format_axis(ax, param_name)
        if add_legend:
            ax.legend(fontsize="x-small", loc="upper right")
        return ax

    @classmethod
    def plot_all_laplace_posteriors_grid(
        cls,
        lfits: Dict[str, "LaplacePosterior"],  # label -> posterior
        true_params: Optional[Parameters] = None,
        prior_mu: Optional[Parameters] = None,
        prior_sigma: Optional[Parameters] = None,
        skip_labels: Iterable[str] = ("ideal",),
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        ax: Optional[plt.Axes] = None,
        color: Optional[str] = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        ncols: int = 2,
        fig_size=(12, 14),
        prior_pdf_interval: Optional[Tuple[float, float]] = None,
        add_legend: bool = True
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
        fig, axes2d, axes = _coerce_axes_grid(ax, nrows, ncols, fig_size)

        for i, name in enumerate(param_names):
            ax = axes[i]
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_title(name)
            ax.set_yticks([])

            # Prior once
            if prior_mu is not None and prior_sigma is not None:
                mu0 = prior_mu.get_from_iterator_name(name)
                sigma0 = prior_sigma.get_from_iterator_name(name)
                cls.plot_prior_only(
                    ax=ax,
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    interval=prior_pdf_interval,
                    label="Prior (log-normal)",
                    color="black",
                    linewidth=2
                )
                # prior = cls._prior_dist_for(mu0, sigma0)
                # q_lo, q_hi = prior_pdf_interval or (0.01, 0.99)
                # xp = np.linspace(prior.ppf(q_lo), prior.ppf(q_hi), 500)
                # yp = prior.pdf(xp)
                # (line,) = ax.plot(xp, yp, label="Prior (log-normal)", color="black", linewidth=2)
                # line_color = line.get_color()
                # y_prior = prior.pdf(mu0)
                # mk = {"marker": "s", "color": line_color, "s": 30, "zorder": 5, "label": "None"}
                # ax.scatter([mu0], [y_prior], **mk)

            # TRUE once
            if true_params is not None:
                true_val = true_params.get_from_iterator_name(name)
                ax.axvline(true_val, color="red", linestyle="--", label="TRUE", linewidth=1)

            # Plot each posterior
            for lbl, lfit in lfits.items():
                if lbl in set(skip_labels):
                    continue
                cls.plot_single_laplace_posterior(
                    param_name=name,
                    lfit=lfit,
                    ax=ax,
                    label=lbl,
                    distribution_type=distribution_type,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    pdf_interval=prior_pdf_interval,
                    show_map_marker=True,
                    marker_kwargs={"label": None},  # avoid duplicate legend entries
                    add_legend=False,
                )

            cls._format_axis(ax, name)
            if add_legend:
                ax.legend(fontsize="x-small", loc="upper right")

        # hide extras
        for j in range(len(param_names), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Laplace Posteriors", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        return fig, axes

import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union, Tuple, Literal
import torch

import numpy as np
import pandas as pd
from math import ceil, erf, sqrt


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, EngFormatter

from typing import Dict, Iterable, Optional, Literal, Tuple, Union
from scipy.stats import lognorm, norm
from pinn_buck.constants import ParameterConstants
import itertools

from typing import Protocol, runtime_checkable

from .laplace_posterior_fitting import LaplacePosterior
from .config import Parameters
from .plot_aux import place_shared_legend


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
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union, Tuple

from .constants import MeasurementGroupArchive
from .laplace_posterior_fitting import LaplacePosterior, LaplacePosteriorConstants


class LaplaceDictionaryLoader:
    """
    Load {label -> LaplacePosterior} from a directory of JSON files.

    - Only files that END with '.laplace.json' are considered (by default).
    - Labels are derived from the filename with the final suffix removed:
        e.g. 'laplace_posterior_10 noise.laplace.json' → 'laplace_posterior_10 noise'
    - Integer labels can be resolved via `group_number_dict` (e.g., MeasurementGroupArchive).
    """

    DEFAULT_MEASUREMENT_GROUP = MeasurementGroupArchive.SHUAI_ORIGINAL
    DEFAULT_FILE_PATTERN = LaplacePosteriorConstants.DEFAULT_FILE_PATTERN

    def __init__(
        self,
        group_number_dict: Optional[Mapping[int, str]] = None,
        file_pattern: Optional[str] = None,
    ) -> None:
        self.group_number_dict: Dict[int, str] = dict(
            group_number_dict or self.DEFAULT_MEASUREMENT_GROUP
        )
        self.FILE_PATTERN: str = file_pattern or self.DEFAULT_FILE_PATTERN

    # ---------- small helpers ----------

    @staticmethod
    def _base_stem(p: Path, final_suffix: str = ".laplace.json") -> str:
        """
        Return the filename with the *final* known suffix stripped.
        If the filename ends with `final_suffix`, strip it; otherwise fall back to Path.stem.
        """
        name = p.name
        if name.lower().endswith(final_suffix.lower()):
            return name[: -len(final_suffix)]
        return p.stem  # fallback (shouldn't happen if globbed with correct pattern)

    def _resolve_labels(self, labels: Iterable[Union[int, str]]) -> List[str]:
        """
        Resolve a mixed list of int/str labels into strings using self.group_number_dict for ints.
        """
        out: List[str] = []
        for lab in labels:
            if isinstance(lab, int):
                if lab not in self.group_number_dict:
                    raise KeyError(f"Integer label {lab} not in group_number_dict.")
                out.append(self.group_number_dict[lab])
            else:
                out.append(lab)
        return out

    def _seen_stems(self, outdir: Path) -> List[str]:
        """
        List of cleaned base stems (without the '.laplace.json' tail) for all matching files.
        """
        files = sorted(outdir.glob(self.FILE_PATTERN))
        return [self._base_stem(p) for p in files]

    @staticmethod
    def _candidate_labels_for(label: str) -> List[str]:
        """
        Generate robust, case-insensitive label variants for matching.
        """
        return list(
            {
                label,
                label.replace(" ", "_"),
                label.replace(" ", "-"),
                re.sub(r"[\s_-]+", " ", label).strip(),
            }
        )

    @staticmethod
    def _match_candidates(label_variants: Sequence[str], base_stem: str) -> bool:
        """
        Decide if a base_stem matches any of the label variants (exact, endswith, or contains).
        """
        base_l = base_stem.lower()
        lowered_exact = {v.lower() for v in label_variants}
        if base_l in lowered_exact:
            return True
        for v in lowered_exact:
            if base_l.endswith(v) or (v in base_l):
                return True
        return False

    # ---------- loaders ----------
    def load_lfits_from_dir(
        self,
        directory: Union[str, Path],
        labels: Optional[Iterable[Union[int, str]]] = None,
        *,
        device: Union[str, "torch.device"] = "cpu",
        raise_error_if_label_not_found: bool = True,
    ) -> Dict[str, "LaplacePosterior"]:
        """
        Load LaplacePosterior JSONs from a directory into {label_str: LaplacePosterior},
        where `label_str` comes from self.group_number_dict (values).
        Preserves the ordering of keys in group_number_dict if labels is None or ints.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        all_files = list(sorted(directory.glob(self.FILE_PATTERN)))
        if not all_files:
            raise FileNotFoundError(f"No files matching '{self.FILE_PATTERN}' in {directory}")

        # Build table of (clean_base_stem -> Path)
        base_index: Dict[str, Path] = {self._base_stem(p): p for p in all_files}
        all_base_stems = list(base_index.keys())

        # Figure out the normalized label list
        if labels is None:
            # default: use *all* group_number_dict entries in key order
            wanted_labels = [
                self.group_number_dict[k] for k in sorted(self.group_number_dict.keys())
            ]
        else:
            wanted_labels = self._resolve_labels(labels)

        result: Dict[str, "LaplacePosterior"] = {}

        for lab in wanted_labels:
            variants = self._candidate_labels_for(lab)
            matches = [bs for bs in all_base_stems if self._match_candidates(variants, bs)]

            if not matches:
                if raise_error_if_label_not_found:
                    raise FileNotFoundError(f"No Laplace JSON found in '{directory}' for label '{lab}'. "
                          f"Available labels: {sorted(all_base_stems)}")
                continue

            unique_matches = list(dict.fromkeys(matches))  # drop dups
            if len(unique_matches) > 1:
                raise FileExistsError(f"Ambiguous files for label '{lab}': {unique_matches}")

            chosen_base = unique_matches[0]
            json_path = base_index[chosen_base]
            # IMPORTANT: use lab (string from group_number_dict) as key
            result[lab] = LaplacePosterior.load(json_path, device=device)

        return result

    def load_one_lfit(
        self,
        directory: Union[str, Path],
        label: Union[int, str],
        *,
        device: Union[str, "torch.device"] = "cpu",
    ) -> "LaplacePosterior":
        """
        Convenience: load a single LaplacePosterior by (approx) label.
        """
        loaded = self.load_lfits_from_dir(
            directory=directory,
            labels=[label],
            device=device,
            raise_error_if_label_not_found=True,
        )
        # return the only value
        return next(iter(loaded.values()))


class LaplacePosteriorPlotter(LaplaceDictionaryLoader):
    """
    Plotter for LaplacePosterior objects.
    Relies on LaplacePosterior.gaussian_approx and .lognormal_approx
    (already in physical units), never on raw theta_log/Sigma_log.
    """

    def __init__(
        self,
        lfits: Dict[str, "LaplacePosterior"]
    ):
        self.lfits = lfits

    @classmethod
    def from_dir(
        cls, 
        directory: Union[str, Path], 
        labels: Optional[List[Union[int, str]]] = None,
        group_number_dict: Optional[Mapping[int, str]] = None
        ) -> "LaplacePosteriorPlotter":
        """
        Load LaplacePosterior objects from a directory.
        """
        raise_error_if_label_not_found = False if (labels is None and group_number_dict is None) else True
        lfits = LaplaceDictionaryLoader(group_number_dict).load_lfits_from_dir(directory, labels=labels, raise_error_if_label_not_found=raise_error_if_label_not_found)
        return cls(lfits)

    @staticmethod
    def _safe_color_generator():
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

    @staticmethod
    def _prior_dist_for(mu, sigma) -> RVLike:
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
    def _sigma_to_quantiles(n_sigma: float) -> Tuple[float, float]:
        # Same math as in your LaplacePosterior helper
        alpha = 0.5 * erf(n_sigma / sqrt(2.0))
        return 0.5 - alpha, 0.5 + alpha

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

    @classmethod
    def _format_abs_axis_with_unit(cls, ax: plt.Axes, param_name: str) -> None:
        """
        Absolute-value axis: attach engineering formatter with the correct unit and
        keep ticks sparse to avoid overlap.
        """
        unit = ParameterConstants.UNITS.get_from_iterator_name(param_name)
        if unit:
            ax.xaxis.set_major_formatter(EngFormatter(unit=unit))
        else:
            ax.set_xlabel("Value")
        # keep ticks sparse so labels don't collide
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # ---------- public API ----------
    def ci_dataframe_label(
        self,
        label: str,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        n_sigma: float = 1.0,
        as_percent: bool = True,
    ) -> pd.DataFrame:
        """
        Build a table with mean, lower, upper for each parameter under the chosen
        marginal (Gaussian or LogNormal). If `as_percent=True`, lower/upper are
        returned as % deviations of the mean (mean becomes 0).
        Columns: ['param','mean','lower','upper','minus','plus','label']
        """
        q_lo, q_hi = self._sigma_to_quantiles(n_sigma)

        if label not in self.lfits:
            raise ValueError(f"Label '{label}' not found in lfits.")

        lfit = self.lfits.get(label)

        if distribution_type == "gaussian":
            dists = lfit.gaussian_approx
        elif distribution_type == "log-normal":
            dists = lfit.lognormal_approx
        else:
            raise ValueError("distribution_type must be 'log-normal' or 'gaussian'.")

        rows = []
        for name, dist in zip(lfit.param_names, dists):
            mean = float(dist.mean())
            lower = float(dist.ppf(q_lo))
            upper = float(dist.ppf(q_hi))
            minus = mean - lower
            plus = upper - mean

            if as_percent:
                # express bounds as +/- % of MEAN; mean becomes 0 baseline
                denom = mean if mean != 0 else 1.0
                minus_val = -100.0 * minus / denom  # positive number (magnitude)
                plus_val = +100.0 * plus / denom
            else:
                minus_val = minus
                plus_val = plus

            rows.append({
                "param": name,
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "minus [%]" if as_percent else "minus": minus_val,
                "plus [%]" if as_percent else "plus": plus_val,
                "label": label,
            })

        df = pd.DataFrame(rows)
        return df

    def ci_dataframe(
        self,
        n_sigma: float = 1.0,
        labels: Optional[Union[str, Iterable[str]]] = None,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        as_percent: bool = True,
    ) -> pd.DataFrame:
        """
        Concatenate CI tables for the selected labels (or all labels if None).

        Returns a DataFrame with rows from each label:
        columns: ['param','mean','lower','upper','minus [%]/minus','plus [%]/plus','label']
        """
        if labels is None:
            labels_to_use = list(self.lfits.keys())
        elif isinstance(labels, str):
            labels_to_use = [labels]
        else:
            labels_to_use = list(labels)

        # sanity check
        missing = [lbl for lbl in labels_to_use if lbl not in self.lfits]
        if missing:
            raise ValueError(f"Labels not found in lfits: {missing}")

        frames = [
            self.ci_dataframe_label(
                label=lbl,
                distribution_type=distribution_type,
                n_sigma=n_sigma,
                as_percent=as_percent,
            )
            for lbl in labels_to_use
        ]
        return pd.concat(frames, ignore_index=True)

    def plot_uncertainty_percent(
        self,
        n_sigma: float = 1.0,
        labels: Optional[Iterable[str]] = None,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        true_params: Optional[Parameters] = None,
        *,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        bar_height: float = 0.35,
        mean_tick_height_factor: float = 0.6,
        colors: Optional[Dict[str, str]] = None,
        title: Optional[str | None] = "Estimation Uncertainty",
        legend: bool = True,
    ):
        """
        Plot horizontal CI segments on a common percentage x-axis.
        Each parameter's mean is mapped to 0; the CI is shown as [-%, +%].
        If `true_params` is provided, a red dashed line marks the TRUE % error
        w.r.t. the *absolute mean of the first label* for that parameter.
        """

        # Labels to include
        if labels is None:
            labels_to_use = list(self.lfits.keys())
        else:
            labels_to_use = list(labels)

        # Validate labels
        missing = [lbl for lbl in labels_to_use if lbl not in self.lfits]
        if missing:
            raise ValueError(f"Labels not found in self.lfits: {missing}")

        # Concatenate CI tables in PERCENT mode
        df = self.ci_dataframe(
            labels=labels_to_use,
            distribution_type=distribution_type,
            n_sigma=n_sigma,
            as_percent=True,  # <<< key: we want percent CI here
        )

        # Parameter order from the first requested label
        ref_label = labels_to_use[0]
        param_names = list(self.lfits[ref_label].param_names)

        # Axes
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure

        # Colors (safe; red reserved for TRUE)
        if colors is None:
            cyc = self._safe_color_generator()
            colors = {lbl: next(cyc) for lbl in labels_to_use}

        # Row layout
        n_params = len(param_names)
        base_y = np.arange(n_params)[::-1]
        offsets = (
            np.linspace(
                -(bar_height * (len(labels_to_use) - 1)) / 2,
                +(bar_height * (len(labels_to_use) - 1)) / 2,
                len(labels_to_use),
            )
            if len(labels_to_use) > 1
            else np.array([0.0])
        )

        # Draw TRUE % lines if requested
        if true_params is not None:
            for i, pname in enumerate(param_names):
                # compute TRUE % error w.r.t. absolute mean of the FIRST label
                idx = self.lfits[ref_label].param_names.index(pname)
                if distribution_type == "gaussian":
                    ref_abs_mean = float(self.lfits[ref_label].gaussian_approx[idx].mean())
                else:
                    ref_abs_mean = float(self.lfits[ref_label].lognormal_approx[idx].mean())
                denom = ref_abs_mean if ref_abs_mean != 0 else 1.0
                true_abs = true_params.get_from_iterator_name(pname)
                x_true = 100.0 * (true_abs - ref_abs_mean) / denom
                ax.axvline(x_true, color="red", linestyle="--", linewidth=1, zorder=0)

        # In percent-mode, df contains:
        # 'minus [%]' and 'plus [%]' (magnitudes), while 'mean','lower','upper' are absolute.
        # For plotting we want:
        #   mean_pct = 0
        #   lo_pct   = -minus[%]
        #   hi_pct   = +plus[%]

        for j, lbl in enumerate(labels_to_use):
            col = colors[lbl]
            sub = df[df["label"] == lbl]
            for i, pname in enumerate(param_names):
                row = sub[sub["param"] == pname]
                if row.empty:
                    continue
                r = row.iloc[0]
                y = base_y[i] + offsets[j]

                lo_pct = float(r["minus [%]"])
                hi_pct = float(r["plus [%]"])
                mean_pct = 0.0

                # CI segment (percent)
                ax.hlines(y, lo_pct, hi_pct, color=col, linewidth=3, label=None, zorder=2)
                # mean tick at 0
                tick_h = bar_height * mean_tick_height_factor / 2
                ax.vlines(
                    mean_pct, y - tick_h, y + tick_h, color=col, linewidth=2, label=None, zorder=3
                )

            # Legend handle per label
            ax.plot([], [], color=col, linewidth=3, label=lbl)

        # Y ticks = parameter names
        ax.set_yticks(base_y)
        ax.set_yticklabels(param_names)

        # Grid and labels
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Deviation from mean [%]")
        if title:
            ax.set_title(title)
        if legend:
            ax.legend(loc="best", frameon=False)

        return fig, ax

    def plot_ci(
        self,
        labels: Optional[Iterable[str]] = None,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        n_sigma: float = 1.0,
        true_params: Optional[Parameters] = None,
        *,
        ncols: int = 2,
        figsize: Tuple[int, int] = (12, 8),
        bar_height: float = 0.35,
        mean_tick_height_factor: float = 0.6,
        colors: Optional[Dict[str, str]] = None,
        title: Optional[str | None] = None,
        legend_title: str = "Labels:",
        legend_ncol: Optional[int] = None,
        legend_frameon: bool = True,
        legend_title_fontsize: Optional[int] = None,
        legend_bbox_to_anchor_horizontal: Optional[float] = 0.5,
        legend_bbox_to_anchor_vertical: Optional[float] = -0.15,
    ):
        """
        One subplot per parameter; inside each subplot, draw one CI bar per label.
        If `as_percentage=True`, all subplots share a common percentage x-axis (means at 0).
        Otherwise, each subplot uses its physical units (independent x-axes).
        """
        # ----- select labels -----
        labels_to_use = list(self.lfits.keys()) if labels is None else list(labels)
        missing = [lbl for lbl in labels_to_use if lbl not in self.lfits]
        if missing:
            raise ValueError(f"Labels not found in self.lfits: {missing}")

        # ----- CI dataframe -----
        df = self.ci_dataframe(
            labels=labels_to_use,
            distribution_type=distribution_type,
            n_sigma=n_sigma,
        )

        ref_label = labels_to_use[0]
        param_names = list(self.lfits[ref_label].param_names)
        n_params = len(param_names)

        # ----- figure & axes -----
        nrows = ceil(n_params / ncols)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, sharex=False, constrained_layout=True
        )
        axes = np.atleast_1d(axes).ravel()

        # ----- colors -----
        if colors is None:
            cyc = self._safe_color_generator()
            colors = {lbl: next(cyc) for lbl in labels_to_use}

        minus_col = "minus"
        plus_col = "plus"

        used_axes = []

        # ----- draw per-parameter -----
        for idx, pname in enumerate(param_names):
            ax = axes[idx]
            used_axes.append(ax)

            ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.set_title(pname)

            # y labels are the labels themselves (top-to-bottom)
            y_positions = np.arange(len(labels_to_use))[::-1]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels_to_use[::-1])

            # TRUE reference (label only on first subplot so shared legend picks it up once)
            if true_params is not None:
                x_true = true_params.get_from_iterator_name(pname)
                ax.axvline(
                    x_true,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label=("TRUE" if idx == 0 else None),
                    zorder=0,
                )

            sub = df[df["param"] == pname]

            # bars for each label
            for j, lbl in enumerate(labels_to_use):
                col = colors[lbl]
                row = sub[sub["label"] == lbl]
                if row.empty:
                    continue
                r = row.iloc[0]

                lo = float(r["lower"])
                hi = float(r["upper"])
                mean_x = float(r["mean"])

                y = y_positions[j]

                # CI segment — set label only on first subplot so legend is shared/clean
                ax.hlines(
                    y, lo, hi, color=col, linewidth=3, zorder=2, label=(lbl if idx == 0 else None)
                )
                # mean tick
                tick_h = bar_height * mean_tick_height_factor / 2
                ax.vlines(mean_x, y - tick_h, y + tick_h, color=col, linewidth=2, zorder=3)

                if idx == 0: 
                    ax.legend(loc="best")

            # axes labels and limits
            # absolute mode → attach the correct physical unit and engineering formatter
            self._format_abs_axis_with_unit(ax, pname)

        # hide any unused axes
        for k in range(len(param_names), len(axes)):
            axes[k].axis("off")

        # ----- shared legend (single, consolidated) -----
        place_shared_legend(
            fig,
            axes,
            legend_title=legend_title,
            frameon=legend_frameon,
            title_fontsize=legend_title_fontsize,
            ncol=legend_ncol,
            bbox_to_anchor_horizontal=legend_bbox_to_anchor_horizontal,
            bbox_to_anchor_vertical=legend_bbox_to_anchor_vertical,
            empty_slot_only=True,  
        )

        if title is None:
            title = f"Laplace Posterior CI Bars [{n_sigma}σ]"
        if title != "":
            fig.suptitle(title, fontsize=14)

        return fig, axes

    # def plot_ci_bars_percent_subplots(
    #     self,
    #     labels: Optional[Iterable[str]] = None,
    #     distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
    #     n_sigma: float = 1.0,
    #     true_params: Optional[Parameters] = None,
    #     *,
    #     ax: Optional[plt.Axes] = None,
    #     figsize: Tuple[int, int] = (10, 3),
    #     bar_height: float = 0.35,
    #     mean_tick_height_factor: float = 0.6,
    #     colors: Optional[Dict[str, str]] = None,
    #     title_prefix: Optional[str] = None,
    #     legend: bool = False,
    # ):
    #     """
    #     Plot one horizontal CI bar chart *per label* on its own subplot.
    #     Each subplot has a common percentage x-axis, mean at 0.
    #     """
    #     import matplotlib.pyplot as plt

    #     # Labels to include
    #     if labels is None:
    #         labels_to_use = list(self.lfits.keys())
    #     else:
    #         labels_to_use = list(labels)

    #     n_labels = len(labels_to_use)

    #     if ax is None:
    #         fig, axes = plt.subplots(
    #             n_labels, 1, figsize=(figsize[0], figsize[1] * n_labels),
    #             constrained_layout=True, sharex=True, sharey=True
    #         )
    #     else:
    #         # check the number of axes is enough
    #         naxes = len(np.atleast_1d(ax).flatten())
    #         if naxes < n_labels:
    #             raise ValueError(f"Not enough axes provided: {naxes} < {n_labels}")
    #         axes = np.atleast_1d(ax).flatten()[:n_labels]  # use only the first n_labels axes
    #         fig = axes[0].figure  # take the figure from the first Axes

    #     # Handle case of single label (axes not an array)
    #     if n_labels == 1:
    #         axes = [axes]

    #     for ax, lbl in zip(axes, labels_to_use):
    #         self.plot_ci_bars_percent(
    #             labels=[lbl],
    #             distribution_type=distribution_type,
    #             n_sigma=n_sigma,
    #             true_params=true_params,
    #             ax=ax,  # pass subplot axis
    #             figsize=figsize,
    #             bar_height=bar_height,
    #             mean_tick_height_factor=mean_tick_height_factor,
    #             colors=colors,
    #             title=(f"{title_prefix}: {lbl}" if title_prefix else lbl),
    #             legend=legend,
    #         )

    #     return fig, axes

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

import re
import pandas as pd 
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union, Literal, Tuple
import torch
import numpy as np

import matplotlib.pyplot as plt

from .laplace_posterior_plotting import LaplacePosteriorPlotter, LaplaceDictionaryLoader
from .constants import MeasurementGroupArchive
from .laplace_posterior_fitting import LaplacePosterior
from .plot_aux import place_shared_legend
from .parameters.parameter_class import Parameters


Experiment = str
Label = str
PosteriorsDict = Dict[Label, LaplacePosterior]
PosteriorDictionary = Dict[Experiment, PosteriorsDict]


class LaplaceResultsComparer:
    """
    Compare Laplace posteriors across one or two experiments (e.g., FW vs FW&BW).

    Conventions
    -----------
    • Each label corresponds to a single JSON file containing one LaplacePosterior.
    • File matching is robust to spaces/underscores/dashes (like ResultsComparerTwo).
    • Plotting uses your LaplacePosteriorPlotter:
        - plot_all_laplace_posteriors_grid(...) to draw parameter grids
        - plot_single_laplace_posterior(...) to focus on one parameter
    """

    DEFAULT_MEASUREMENT_GROUP = LaplacePosteriorPlotter.DEFAULT_MEASUREMENT_GROUP
    FILE_PATTERN = LaplacePosteriorPlotter.DEFAULT_FILE_PATTERN

    def __init__(
        self,
        posterior_dictionary: Optional[PosteriorDictionary] = None,
        group_number_dict: Optional[Mapping[int, str]] = None,
        file_pattern: Optional[str] = None,
    ) -> None:
        self.group_number_dict = dict(group_number_dict or self.DEFAULT_MEASUREMENT_GROUP)
        self.posterior_dictionary: PosteriorDictionary = posterior_dictionary or {}
        if file_pattern:
            self.FILE_PATTERN = file_pattern
            
        # units per parameter, from the first loaded posterior
        all_units = {}
        for runs in self.posterior_dictionary.values():
            for lfit in runs.values():
                for pname, unit in lfit.param_units.items():
                    if pname in all_units:
                        if all_units[pname] != unit:
                            raise ValueError(f"Conflicting units for parameter '{pname}': '{all_units[pname]}' vs '{unit}'")
                    else:
                        all_units[pname] = unit
        self.units = Parameters(**all_units)    
            
    # ------------ Label helpers ------------
    def _resolve_labels(self, labels: Iterable[Union[int, str]]) -> List[str]:
        return LaplaceDictionaryLoader._resolve_labels(self.group_number_dict, labels)

    def _seen_stems(self, outdir: Path) -> list[str]:
        return [p.stem for p in sorted(outdir.glob(self.FILE_PATTERN))]

    def _load_specific_labels(
        self,
        outdir: Path,
        labels: Iterable[str],
        raise_error_if_label_not_found: bool = True,
        device: Union[str, "torch.device"] = "cpu",
    ) -> PosteriorsDict:
        loader = LaplacePosteriorPlotter(  # we just need the loader behavior
            lfits={}
        )
        # leverage the class it inherits from (LaplaceDictionaryLoader)
        # NOTE: we want to respect group_number_dict mapping (ints → display names)
        # so pass self.group_number_dict into a dedicated loader instance:
        tmp_loader = LaplaceDictionaryLoader(group_number_dict=self.group_number_dict)
        return tmp_loader.load_lfits_from_dir(
            directory=outdir,
            labels=labels,
            device=device,
            raise_error_if_label_not_found=raise_error_if_label_not_found,
        )

    def _normalize_experiments(
        self,
        experiments: Optional[Union[str, Tuple[str, str]]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalize experiments input:
        - None: if exactly 2 in dict, return (expA, expB); if 1, return (expA, None)
        - "Exp": return (Exp, None)
        - ("A","B"): return as-is
        """
        keys = list(self.posterior_dictionary.keys())
        if experiments is None:
            if len(keys) == 2:
                return keys[0], keys[1]
            elif len(keys) == 1:
                return keys[0], None
            else:
                raise ValueError(
                    "Please pass `experiments` when there are not 1 or 2 experiments. "
                    f"Available: {sorted(keys)}"
                )
        if isinstance(experiments, str):
            if experiments not in self.posterior_dictionary:
                raise KeyError(f"Experiment '{experiments}' not found. Available: {sorted(keys)}")
            return experiments, None
        if isinstance(experiments, tuple) and len(experiments) == 2:
            a, b = experiments
            for e in (a, b):
                if e not in self.posterior_dictionary:
                    raise KeyError(f"Experiment '{e}' not found. Available: {sorted(keys)}")
            return a, b
        raise TypeError("experiments must be None, a single string, or a (str, str) tuple.")

    # ------------ Builders ------------
    @classmethod
    def from_dirs(
        cls,
        exp_dirs: Mapping[Experiment, Union[str, Path]],
        labels: Optional[Iterable[Union[int, str]]] = None,
        **kwargs,
    ) -> "LaplaceResultsComparer":
        for exp, d in exp_dirs.items():
            if not Path(d).is_dir():
                raise FileNotFoundError(f"Directory not found: {d}")

        tmp = cls(**kwargs)
        if labels is None:
            resolved = list(tmp.group_number_dict.values())
            missing_ok = True
        else:
            resolved = tmp._resolve_labels(labels)
            missing_ok = False

        post_dict: PosteriorDictionary = {}
        for exp, d in exp_dirs.items():
            runs = tmp._load_specific_labels(
                Path(d), resolved, raise_error_if_label_not_found=not missing_ok
            )
            if runs:
                post_dict[exp] = runs
        if not post_dict:
            raise ValueError("No experiments loaded; check directories and labels.")
        return cls(
            posterior_dictionary=post_dict,
            group_number_dict=tmp.group_number_dict,
            file_pattern=tmp.FILE_PATTERN,
        )

    def ci_table(
        self,
        experiments: Optional[Union[str, Tuple[str, str]]] = None,
        labels: Optional[Iterable[Union[int, str]]] = None,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        n_sigma: float = 1.0,
    ) -> pd.DataFrame:
        """
        Return a concatenated CI table across experiments with an 'experiment' column.
        Columns are those from LaplacePosteriorPlotter.ci_dataframe(...) plus 'experiment'.
        """
        expA, expB = self._normalize_experiments(experiments)
        exps = [expA] + ([expB] if expB is not None else [])

        frames = []
        for exp in exps:
            runs = self.posterior_dictionary[exp]
            # select labels in this experiment
            if labels is None:
                use_labels = list(runs.keys())
            else:
                use_labels = self._resolve_labels(labels)
            plotter = LaplacePosteriorPlotter(runs)
            df = plotter.ci_dataframe(
                labels=use_labels,
                distribution_type=distribution_type,
                n_sigma=n_sigma,
            )
            df.insert(0, "experiment", exp)
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    # ------------ Plotters ------------
    def plot_ci(
        self,
        experiments: Optional[Union[str, Tuple[str, str]]] = None,
        labels: Optional[Iterable[Union[int, str]]] = None,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        n_sigma: float = 1.0,
        *,
        ncols: int = 2,
        skip_labels: Tuple[str, ...] = (),
        figsize: Tuple[int, int] = (12, 8),
        bar_height: float = 0.35,
        mean_tick_height_factor: float = 0.6,
        colors: Optional[Dict[str, str]] = None,  # colors per label
        linestyles: Optional[Mapping[str, str]] = None,  # linestyle per experiment
        true_params: Optional[Parameters] = None,
        keep_same_color_for_different_experiments: bool = False,
        legend: bool = True,
        legend_title: str = "Parameters:",
        legend_ncol: Optional[int] = None,
        legend_frameon: bool = True,
        legend_bottom_inch: float = 0.15,
    ):
        """
        One subplot per parameter. For each label (row), draw one CI bar per experiment,
        placed close to each other (y-offsets). Color encodes label; linestyle encodes experiment.
        """
        # ----- data -----
        df = self.ci_table(
            experiments=experiments,
            labels=labels,
            distribution_type=distribution_type,
            n_sigma=n_sigma,
        )
        expA, expB = self._normalize_experiments(experiments)
        exps = [expA] + ([expB] if expB is not None else [])

        # param order from first experiment's first label
        first_exp = exps[0]
        first_label = df.loc[df["experiment"] == first_exp, "label"].iloc[0]
        param_names = list(df[(df["experiment"] == first_exp) & (df["label"] == first_label)]["param"])
        n_params = len(param_names)

        # labels order from the intersection present for all exps
        if labels is None:
            labels_all = sorted(df["label"].unique())
        else:
            labels_all = list(self._resolve_labels(labels))
        # keep only labels that appear for *all* experiments
        labels_to_use = [lbl for lbl in labels_all if set(exps).issubset(set(df[df["label"]==lbl]["experiment"]))]
        labels_to_use = [lbl for lbl in labels_to_use if lbl not in skip_labels]

        # ----- figure & axes -----
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, sharex=False, sharey=True, constrained_layout=True
        )
        axes = np.atleast_1d(axes).ravel()

        # ----- styling: colors per label, linestyles per experiment -----
        if colors is None:
            cyc = LaplacePosteriorPlotter._safe_color_generator()
            if keep_same_color_for_different_experiments:
                colors = {lbl: next(cyc) for lbl in labels_to_use}
            else:
                colors = {lbl: {exp: next(cyc) for exp in exps} for lbl in labels_to_use}

        if linestyles is None:
            # match your posterior grid defaults
            ls_default = ["-", "-"]
            linestyles = {exp: ls_default[i % len(ls_default)] for i, exp in enumerate(exps)}

        # offsets: cluster bars for each label
        y_base = np.arange(len(labels_to_use))[::-1]  # top-to-bottom
        k = len(exps)
        if k == 1:
            offsets = np.array([0.0])
        else:
            # spread within a total window comparable to bar_height
            offsets = np.linspace(-bar_height/2, +bar_height/2, k)

        for idx, pname in enumerate(param_names):
            ax = axes[idx]
            ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.set_title(pname)

            # y ticks are labels (just once per subplot)
            ax.set_yticks(y_base)
            ax.set_yticklabels(labels_to_use)

            # TRUE reference (label only on the first subplot for legend)
            if true_params is not None:
                x_true = true_params.get_from_iterator_name(pname)
                ax.axvline(
                    x_true, color="red", linestyle="-", linewidth=1,
                    label="TRUE", zorder=0,
                )

            # subset for this parameter
            sub_p = df[df["param"] == pname]

            # draw per label + experiment
            for j, lbl in enumerate(labels_to_use):
                y0 = y_base[j]
                for e_idx, exp in enumerate(exps):
                    col = colors[lbl] if keep_same_color_for_different_experiments else colors[lbl][exp]
                    row = sub_p[(sub_p["label"] == lbl) & (sub_p["experiment"] == exp)]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    lo = float(r["lower"])
                    hi = float(r["upper"])
                    mean_x = float(r["mean"])
                    y = y0 + offsets[e_idx]
                    label_name = f"{exp}{lbl}"

                    ax.hlines(y, lo, hi, color=col, linestyle=linestyles[exp], linewidth=3, zorder=2, label=label_name)
                    tick_h = bar_height * mean_tick_height_factor / 2
                    ax.vlines(mean_x, y - tick_h, y + tick_h, color=col, linestyle=linestyles[exp], linewidth=2, zorder=3)
                    ax.legend(loc="best")

            # engineering units on x-axis
            LaplacePosteriorPlotter._format_abs_axis_with_unit(ax, pname, self.units)

        # hide unused axes
        for k_ax in range(len(param_names), len(axes)):
            axes[k_ax].axis("off")

            # legend like ResultsComparerTwo
        if legend:
            place_shared_legend(
                fig,
                fig.axes,
                legend_title=legend_title,
                ncol=legend_ncol,
                frameon=legend_frameon,
                legend_bottom_inch=legend_bottom_inch,
            )

        fig.suptitle(f"Laplace Posterior CI Bars [{n_sigma}σ]", fontsize=14)
        return fig, axes

    def plot_posteriors_grid(
        self,
        experiments: Optional[Union[str, Tuple[str, str]]] = None,
        labels: Optional[Iterable[Union[int, str]]] = None,
        true_params: Optional[Parameters] = None,
        prior_mu: Optional[Parameters] = None,
        prior_sigma: Optional[Parameters] = None,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        ncols: int = 2,
        fig_size=(10, 8),
        pdf_interval: Optional[Tuple[float, float]] = None,
        prior_pdf_interval: Optional[Tuple[float, float]] = None,
        legend: bool = True,
        legend_title: Optional[str] = "Posteriors:",
        legend_ncol: Optional[int] = None,
        legend_frameon: bool = True,
        legend_bottom_inch: float = 0.15,
        label_prefixes: Optional[Tuple[str, str]] = None,
        linestyles: Optional[Union[str, Tuple[str, str]]] = None,
        linewidth: float = 1.0,
        color: Optional[str] = None,
        skip_labels: Optional[Tuple[str, ...]] = None,
        keep_same_color_for_different_experiments: bool = False,
    ):
        """
        Grid of PDFs for all parameters.
        • Single experiment: overlays selected labels on one grid.
        • Two experiments: overlays BOTH experiments on the SAME grid (solid vs dashed),
        using the same label set (intersection by default).
        """
        expA, expB = self._normalize_experiments(experiments)
        runsA = self.posterior_dictionary[expA]

        # resolve label set
        if expB is None:
            use_labels = sorted(runsA.keys()) if labels is None else self._resolve_labels(labels)
            use_labels = [lbl for lbl in use_labels if lbl not in (skip_labels or [])]
            lfitsA = {k: runsA[k] for k in use_labels}            
            # reuse your existing grid plotter for single experiment
            plotterA = LaplacePosteriorPlotter(lfitsA)
            fig, axes = plotterA.plot_laplace_posteriors(
                true_params=true_params,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                distribution_type=distribution_type,
                ncols=ncols,
                fig_size=fig_size,
                prior_pdf_interval=prior_pdf_interval,
                pdf_interval=pdf_interval,
                add_legend=legend,
                color=color,
                linewidth=linewidth,
            )
            fig.suptitle(expA, fontsize=16)
            return fig, axes

        # two experiments → overlay on same grid
        runsB = self.posterior_dictionary[expB]
        if labels is None:
            inter = sorted(set(runsA.keys()) & set(runsB.keys()))
            inter = [lbl for lbl in inter if lbl not in (skip_labels or [])]
            if not inter:
                raise ValueError(
                    f"No common labels between '{expA}' and '{expB}'. "
                    f"{expA} keys: {sorted(runsA.keys())}, {expB} keys: {sorted(runsB.keys())}"
                )
            use_labels = inter
        else:
            use_labels = self._resolve_labels(labels)

        # style
        if linestyles is None:
            lsA, lsB = "-", "-."
            linewidthA, linewidthB = linewidth, linewidth
        elif isinstance(linestyles, str):
            lsA = lsB = linestyles
            linewidthA = linewidthB = linewidth
        else:
            if len(linestyles) != 2:
                raise ValueError("linestyles must be a string or a 2-tuple of strings.")
            lsA, lsB = linestyles
            linewidthA = linewidthB = linewidth

        prefixes = label_prefixes or (f"{expA}_", f"{expB}_")

        # figure grid
        # use first posterior to define param order
        ref_lfit = next(iter(runsA.values()))
        param_names = list(ref_lfit.param_names)
        nrows = int(np.ceil(len(param_names) / ncols))
        fig, axes2d = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, constrained_layout=True)
        axes = np.array(axes2d).ravel()

        color_cycle = LaplacePosteriorPlotter._safe_color_generator()

        if keep_same_color_for_different_experiments:
            label_to_color = {lbl: next(color_cycle) for lbl in use_labels}
        else:
            label_to_color = {lbl: {exp: next(color_cycle) for exp in (expA, expB)} for lbl in use_labels}

        plotterA = LaplacePosteriorPlotter(runsA)
        plotterB = LaplacePosteriorPlotter(runsB)

        # draw per-parameter
        for i, name in enumerate(param_names):
            ax = axes[i]
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_title(name)
            ax.set_yticks([])

            # prior + TRUE once
            if prior_mu is not None and prior_sigma is not None:
                mu0 = prior_mu.get_from_iterator_name(name)
                sigma0 = prior_sigma.get_from_iterator_name(name)
                LaplacePosteriorPlotter.plot_prior_only(
                    ax=ax,
                    mu=mu0,
                    sigma=sigma0,
                    interval=(prior_pdf_interval or LaplacePosteriorPlotter.DEFAULT_PDF_INTERVAL),
                    color="black",
                    linewidth=2,
                    linestyle="-",
                    label="Prior",
                    show_marker=True,
                    marker_kwargs={"label": None},  # avoid legend duplication
                )
            if true_params is not None:
                ax.axvline(
                    true_params.get_from_iterator_name(name),
                    color="red",
                    linestyle="-",
                    linewidth=1,
                    label="TRUE",
                )

            # overlay A (solid) and B (dashed)
            for lbl in use_labels:
                # A
                default_color = label_to_color[lbl]
                if keep_same_color_for_different_experiments:
                    colA = colB =color or default_color
                else:
                    colA = default_color[expA]
                    colB = default_color[expB]
                # col = color or color_dict[expA]
                plotterA.plot_single_laplace_posterior(
                    label=lbl,
                    param_name=name,
                    ax=ax,
                    label_name=f"{prefixes[0]}{lbl}",
                    distribution_type=distribution_type,
                    pdf_interval=pdf_interval,
                    color=colA,
                    linestyle=lsA,
                    linewidth=linewidthA,
                    add_legend=False,
                    show_map_marker=True,
                    marker_kwargs={"label": None},
                )
                # B
                plotterB.plot_single_laplace_posterior(
                    label=lbl,
                    param_name=name,
                    ax=ax,
                    label_name=f"{prefixes[1]}{lbl}",
                    distribution_type=distribution_type,
                    pdf_interval=pdf_interval,
                    color=colB,
                    linestyle=lsB,
                    linewidth=linewidthB,
                    add_legend=False,
                    show_map_marker=True,
                    marker_kwargs={"marker": "o", "label": None},
                )

            # units/format
            LaplacePosteriorPlotter._format_abs_axis_with_unit(ax, name, self.units)

        # hide extra axes
        for j in range(len(param_names), len(axes)):
            axes[j].axis("off")

        # legend like ResultsComparerTwo
        if legend:
            place_shared_legend(
                fig,
                fig.axes,
                legend_title=legend_title,
                ncol=legend_ncol,
                frameon=legend_frameon,
                legend_bottom_inch=legend_bottom_inch,
            )

        fig.suptitle(f"{expA} (solid) vs {expB} (dashed)", fontsize=16)
        return fig, axes

    def plot_param_overlay(
        self,
        param_name: str,
        experiments: Optional[Union[str, Tuple[str, str]]] = None,
        labels: Optional[Iterable[Union[int, str]]] = None,
        distribution_type: Literal["log-normal", "gaussian"] = "log-normal",
        true_params: Optional[Parameters] = None,
        prior_mu: Optional[Parameters] = None,
        prior_sigma: Optional[Parameters] = None,
        prior_pdf_interval: Optional[Tuple[float, float]] = None,
        color: Optional[str] = None,
        linestyle: str = "-",
        linewidth: float = 1.0,
        figsize=(7, 4),
        legend: bool = True,
        label_prefixes: Optional[Tuple[str, str]] = None,
        **kwargs,
    ):
        """
        Overlay the chosen parameter's PDF across labels.
        • Single experiment: one axis with multiple label curves.
        • Two experiments: side-by-side axes with the same label set.
        """
        expA, expB = self._normalize_experiments(experiments)
        runsA = self.posterior_dictionary[expA]

        prefixes = label_prefixes or (f"{expA}_", f"{expB}_")

        plotterA = LaplacePosteriorPlotter(runsA)
        if expB is not None:
            runsB = self.posterior_dictionary[expB]
            plotterB = LaplacePosteriorPlotter(runsB)

        # label set
        if expB is None:
            use_labels = sorted(runsA.keys()) if labels is None else self._resolve_labels(labels)
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
            # prior + truth once
            if prior_mu is not None and prior_sigma is not None:
                mu0 = prior_mu.get_from_iterator_name(param_name)
                sigma0 = prior_sigma.get_from_iterator_name(param_name)
                LaplacePosteriorPlotter.plot_prior_only(
                    ax=ax,
                    mu=mu0,
                    sigma=sigma0,
                    interval=prior_pdf_interval,
                )
            if true_params is not None:
                ax.axvline(
                    true_params.get_from_iterator_name(param_name),
                    color="red",
                    linestyle="--",
                    label="TRUE",
                    linewidth=1,
                )

            # overlays
            for lbl in use_labels:
                plotterA.plot_single_laplace_posterior(
                    param_name=param_name,
                    label=lbl,
                    label_name=f"{prefixes[0]}{lbl}",
                    ax=ax,
                    distribution_type=distribution_type,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    add_legend=False,
                )
            if legend:
                ax.legend(fontsize="small")
            ax.set_title(f"{expA} — {param_name}")
            return fig, ax

        # two experiments
        runsB = self.posterior_dictionary[expB]
        if labels is None:
            inter = sorted(set(runsA.keys()) & set(runsB.keys()))
            if not inter:
                raise ValueError(f"No common labels between '{expA}' and '{expB}'.")
            use_labels = inter
        else:
            use_labels = self._resolve_labels(labels)

        fig, axes = plt.subplots(
            1, 2, figsize=(figsize[0] * 1.5, figsize[1]), constrained_layout=True
        )
        for side, (exp, runs, ax) in enumerate(((expA, runsA, axes[0]), (expB, runsB, axes[1]))):
            # prior + truth once
            if prior_mu is not None and prior_sigma is not None:
                mu0 = prior_mu.get_from_iterator_name(param_name)
                sigma0 = prior_sigma.get_from_iterator_name(param_name)
                LaplacePosteriorPlotter.plot_prior_only(
                    ax=ax,
                    mu=mu0,
                    sigma=sigma0,
                    interval=prior_pdf_interval,
                )

            if true_params is not None:
                ax.axvline(
                    true_params.get_from_iterator_name(param_name),
                    color="red",
                    linestyle="-",
                    label="TRUE",
                    linewidth=1,
                )

            for lbl in use_labels:
                plotter = plotterA if side == 0 else plotterB
                plotter.plot_single_laplace_posterior(
                    param_name=param_name,
                    label=lbl,
                    label_name=f"{prefixes[side]}{lbl}",
                    ax=ax,
                    distribution_type=distribution_type,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    add_legend=False,
                )
            if legend:
                ax.legend(fontsize="small")
            ax.set_title(f"{exp} — {param_name}")
        return fig, axes

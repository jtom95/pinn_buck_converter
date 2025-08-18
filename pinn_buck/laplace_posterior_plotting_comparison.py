import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union, Literal, Tuple
import torch
import numpy as np

import matplotlib.pyplot as plt

from .laplace_posterior_plotting import LaplacePosteriorPlotter
from .constants import MeasurementGroupArchive
from .laplace_posterior_fitting import LaplacePosterior
from .plot_aux import place_shared_legend
from .config import Parameters


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

    DEFAULT_MEASUREMENT_GROUP = MeasurementGroupArchive.SHUAI_ORIGINAL

    # Default file pattern; override if your saved files differ
    FILE_PATTERN = "*.laplace.json"

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

    # ------------ Label helpers ------------
    def _resolve_labels(self, labels: Iterable[Union[int, str]]) -> List[str]:
        out: List[str] = []
        for lab in labels:
            if isinstance(lab, int):
                if lab not in self.group_number_dict:
                    raise KeyError(f"Integer label {lab} not in group_number_dict.")
                out.append(self.group_number_dict[lab])
            else:
                out.append(lab)
        return out

    def _seen_stems(self, outdir: Path) -> list[str]:
        return [p.stem for p in sorted(outdir.glob(self.FILE_PATTERN))]

    def _load_specific_labels(
        self,
        outdir: Path,
        labels: Iterable[str],
        raise_error_if_label_not_found: bool = True,
        device: Union[str, "torch.device"] = "cpu",
    ) -> PosteriorsDict:
        out: PosteriorsDict = {}
        all_files = list(sorted(outdir.glob(self.FILE_PATTERN)))
        stems = [f.stem for f in all_files]

        def candidates_for(label: str) -> list[Path]:
            label_variants = {
                label,
                label.replace(" ", "_"),
                label.replace(" ", "-"),
                re.sub(r"[\s_-]+", " ", label).strip(),
            }
            cands: list[Path] = []
            lowered_exact = {v.lower() for v in label_variants}
            for f in all_files:
                if f.stem.lower() in lowered_exact:
                    cands.append(f)
            if cands:
                return cands
            lowered = [v.lower() for v in label_variants]
            for f in all_files:
                stem_l = f.stem.lower()
                if any(stem_l.endswith(v) or v in stem_l for v in lowered):
                    cands.append(f)
            return cands

        for label in labels:
            cands = candidates_for(label)
            if not cands:
                if raise_error_if_label_not_found:
                    raise FileNotFoundError(
                        f"No Laplace JSON found in '{outdir}' for label '{label}'. "
                        f"Available stems: {stems}"
                    )
                continue
            unique_stems = {c.stem for c in cands}
            if len(unique_stems) > 1:
                raise FileExistsError(
                    f"Ambiguous files for label '{label}': {[c.name for c in cands]}"
                )
            json_file = cands[0]
            lfit = LaplacePosterior.load(json_file, device=device)
            out[label] = lfit
        return out

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
    def from_dirs_two(
        cls,
        exp1: Experiment,
        dir1: Union[str, Path],
        exp2: Experiment,
        dir2: Union[str, Path],
        labels: Optional[Iterable[Union[int, str]]] = None,
        **kwargs,
    ) -> "LaplaceResultsComparer":
        tmp = cls(**kwargs)
        if labels is None:
            resolved = list(tmp.group_number_dict.values())
            missing_ok = True
        else:
            resolved = tmp._resolve_labels(labels)
            missing_ok = False

        runs1 = tmp._load_specific_labels(
            Path(dir1), resolved, raise_error_if_label_not_found=not missing_ok
        )
        runs2 = tmp._load_specific_labels(
            Path(dir2), resolved, raise_error_if_label_not_found=not missing_ok
        )

        if not runs1:
            raise ValueError(f"No posteriors loaded for '{exp1}' from {dir1}.")
        if not runs2:
            raise ValueError(f"No posteriors loaded for '{exp2}' from {dir2}.")

        return cls(
            posterior_dictionary={exp1: runs1, exp2: runs2},
            group_number_dict=tmp.group_number_dict,
            file_pattern=tmp.FILE_PATTERN,
        )

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

    # ------------ Plotters ------------
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
        legend_frameon: bool = False,
        legend_bbox_to_anchor_vertical: float = -0.1,
        label_prefixes: Optional[Tuple[str, str]] = None,
        linestyles: Optional[Union[str, Tuple[str, str]]] = None,
        linewidth: float = 1.,
        color: Optional[str] = None,
        skip_labels: Optional[Tuple[str, ...]] = None,
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
            fig, axes = LaplacePosteriorPlotter.plot_all_laplace_posteriors_grid(
                lfits=lfitsA,
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
            lsA, lsB = "-", "--"
            linewidthA, linewidthB = linewidth + 0.5, linewidth
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
        label_to_color = {lbl: {exp: next(color_cycle) for exp in (expA, expB)} for lbl in use_labels}

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
                    interval=(prior_pdf_interval or (0.01, 0.99)),
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
                color_dict = label_to_color[lbl]
                LaplacePosteriorPlotter.plot_single_laplace_posterior(
                    param_name=name,
                    lfit=runsA[lbl],
                    ax=ax,
                    label=f"{prefixes[0]}{lbl}",
                    distribution_type=distribution_type,
                    pdf_interval=pdf_interval,
                    color=color or color_dict[expA],
                    linestyle=lsA,
                    linewidth=linewidthA,
                    add_legend=False,
                    show_map_marker=True,
                    marker_kwargs={"label": None},
                )
                # B
                LaplacePosteriorPlotter.plot_single_laplace_posterior(
                    param_name=name,
                    lfit=runsB[lbl],
                    ax=ax,
                    label=f"{prefixes[1]}{lbl}",
                    distribution_type=distribution_type,
                    pdf_interval=pdf_interval,
                    color=color or color_dict[expB],
                    linestyle=lsB,
                    linewidth=linewidthB,
                    add_legend=False,
                    show_map_marker=True,
                    marker_kwargs={"marker": "o", "label": None},
                )

            # units/format
            LaplacePosteriorPlotter._format_axis(ax, name)

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
                bbox_to_anchor_vertical=legend_bbox_to_anchor_vertical,
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
        **kwargs,
    ):
        """
        Overlay the chosen parameter's PDF across labels.
        • Single experiment: one axis with multiple label curves.
        • Two experiments: side-by-side axes with the same label set.
        """
        expA, expB = self._normalize_experiments(experiments)
        runsA = self.posterior_dictionary[expA]

        # label set
        if expB is None:
            use_labels = sorted(runsA.keys()) if labels is None else self._resolve_labels(labels)
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
            # prior + truth once
            if prior_mu is not None and prior_sigma is not None:
                mu0 = prior_mu.get_from_iterator_name(param_name)
                sigma0 = prior_sigma.get_from_iterator_name(param_name)
                LaplacePosteriorPlotter.plot_single_laplace_posterior(
                    param_name=param_name,
                    lfit=next(iter(runsA.values())),
                    ax=ax,
                    label=None,
                    distribution_type="log-normal",
                    prior_mu=mu0,
                    prior_sigma=sigma0,
                    prior_pdf_interval=prior_pdf_interval,
                    color="black",
                    linewidth=2,
                    linestyle="-",
                    add_legend=False,
                    show_map_marker=True,
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
                LaplacePosteriorPlotter.plot_single_laplace_posterior(
                    param_name=param_name,
                    lfit=runsA[lbl],
                    ax=ax,
                    label=lbl,
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
                LaplacePosteriorPlotter.plot_single_laplace_posterior(
                    param_name=param_name,
                    lfit=next(iter(runs.values())),
                    ax=ax,
                    label=None,
                    distribution_type="log-normal",
                    prior_mu=mu0,
                    prior_sigma=sigma0,
                    prior_pdf_interval=prior_pdf_interval,
                    color="black",
                    linewidth=2,
                    linestyle="-",
                    add_legend=False,
                    show_map_marker=True,
                )
            if true_params is not None:
                ax.axvline(
                    true_params.get_from_iterator_name(param_name),
                    color="red",
                    linestyle="--",
                    label="TRUE",
                    linewidth=1,
                )

            for lbl in use_labels:
                LaplacePosteriorPlotter.plot_single_laplace_posterior(
                    param_name=param_name,
                    lfit=runs[lbl],
                    ax=ax,
                    label=lbl,
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

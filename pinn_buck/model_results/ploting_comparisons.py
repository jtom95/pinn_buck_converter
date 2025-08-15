import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union, Literal, Tuple

import matplotlib.pyplot as plt

from .history import TrainingHistory  # adjust if your import differs
from .plotting_single_run import (
    plot_tracked_parameters,
    plot_final_percentage_error_multi,
)
from ..plot_aux import place_shared_legend
from ..constants import MeasurementGroupArchive

Experiment = str
Label = str
RunsDict = Dict[Label, TrainingHistory]
RunDictionary = Dict[Experiment, RunsDict]


class ResultsComparerTwo:
    """
    Compare two sets of runs (e.g., forward vs forward&backward).

    Initialize either with prebuilt dicts (label -> TrainingHistory) or
    use `from_dirs(...)` to load only the labels you care about. If `labels`
    is None in `from_dirs`, the class tries all labels from `group_number_dict`
    and silently skips labels with no matching CSV.
    """
    DEFAULT_MEASUREMENT_GROUP = MeasurementGroupArchive.SHUAI_ORIGINAL

    FILE_PATTERN = "*.csv"

    def __init__(
        self,
        run_dictionary: Optional[Dict[Experiment, TrainingHistory]] = None,
        group_number_dict: Optional[Mapping[int, str]] = None,
        drop_columns: Sequence[str] = ("learning_rate",),
    ) -> None:
        self.drop_columns = tuple(drop_columns)
        self.group_number_dict = dict(group_number_dict or self.DEFAULT_MEASUREMENT_GROUP)
        self.run_dictionary: RunsDict = run_dictionary or {}

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
    ) -> RunsDict:
        out: RunsDict = {}
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
                        f"No CSV found in '{outdir}' for label '{label}'. "
                        f"Available stems: {stems}"
                    )
                continue
            unique_stems = {c.stem for c in cands}
            if len(unique_stems) > 1:
                raise FileExistsError(
                    f"Ambiguous files for label '{label}': {[c.name for c in cands]}"
                )
            csv_file = cands[0]
            tr = TrainingHistory.from_csv(csv_file)
            if self.drop_columns:
                tr = tr.drop_columns(list(self.drop_columns))
            out[label] = tr
        return out

    def _get_runs(self, experiment: Experiment) -> RunsDict:
        if experiment not in self.run_dictionary:
            raise KeyError(
                f"Experiment '{experiment}' not found. "
                f"Available: {sorted(self.run_dictionary.keys())}"
            )
        return self.run_dictionary[experiment]

    @classmethod
    def from_dirs_two(
        cls,
        exp1: Experiment,
        dir1: Union[str, Path],
        exp2: Experiment,
        dir2: Union[str, Path],
        labels: Optional[Iterable[Union[int, str]]] = None,
        **kwargs,
    ) -> "ResultsComparerTwo":
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
            raise ValueError(f"No runs loaded for '{exp1}' from {dir1}.")
        if not runs2:
            raise ValueError(f"No runs loaded for '{exp2}' from {dir2}.")

        return cls(
            run_dictionary={exp1: runs1, exp2: runs2},
            group_number_dict=tmp.group_number_dict,
            drop_columns=tmp.drop_columns,
        )

    @classmethod
    def from_dirs(
        cls,
        exp_dirs: Mapping[Experiment, Union[str, Path]],
        labels: Optional[Iterable[Union[int, str]]] = None,
        **kwargs,
    ) -> "ResultsComparerTwo":

        # check the directory existence
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

        run_dictionary: RunDictionary = {}
        for exp, d in exp_dirs.items():
            runs = tmp._load_specific_labels(
                Path(d), resolved, raise_error_if_label_not_found=not missing_ok
            )
            if runs:
                run_dictionary[exp] = runs
        if not run_dictionary:
            raise ValueError("No experiments loaded; check directories and labels.")
        return cls(
            run_dictionary=run_dictionary,
            group_number_dict=tmp.group_number_dict,
            drop_columns=tmp.drop_columns,
        )

    def _subset(self, runs: RunsDict, labels: Iterable[Union[int, str]]) -> RunsDict:
        keys = self._resolve_labels(labels)
        return {k: runs[k] for k in keys}

    def _normalize_experiments(
        self,
        experiments: Optional[Union[str, Tuple[str, str]]]=None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalize experiments input:
        - None: if exactly 2 in dict, return (expA, expB); if 1, return (expA, None)
        - "Exp": return (Exp, None)
        - ("A","B"): return as-is
        """
        keys = list(self.run_dictionary.keys())
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
            if experiments not in self.run_dictionary:
                raise KeyError(f"Experiment '{experiments}' not found. Available: {sorted(keys)}")
            return experiments, None
        if isinstance(experiments, tuple) and len(experiments) == 2:
            a, b = experiments
            for e in (a, b):
                if e not in self.run_dictionary:
                    raise KeyError(f"Experiment '{e}' not found. Available: {sorted(keys)}")
            return a, b
        raise TypeError("experiments must be None, a single string, or a (str, str) tuple.")

    def plot_comparison(
        self,
        experiments: Optional[Union[str, Tuple[str, str]]] = None,
        labels: Optional[Iterable[Union[int, str]]] = None,
        target=None,
        select_lowest_loss: bool = False,
        figsize: Tuple[int, int] = (8, 3),
        suptitle: Optional[str] = "Final Percentage Error",
        titles: Optional[Tuple[str, str]] = None,
        sharey: bool = True,
        legend_bbox_to_anchor_vertical: float = -0.1,
    ):
        expA, expB = self._normalize_experiments(experiments)
        runsA = self.run_dictionary[expA]

        # Single-experiment case → show one panel
        if expB is None:
            # labels default: all in the chosen experiment
            if labels is None:
                use_labels = sorted(runsA.keys())
            else:
                use_labels = self._resolve_labels(labels)
            runs_orderedA = {k: runsA[k] for k in use_labels}

            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
            plot_final_percentage_error_multi(
                runs=runs_orderedA,
                target=target,
                figsize=figsize,
                select_lowest_loss=select_lowest_loss,
                ax=ax,
            )
            ax.set_title((titles[0] if titles else expA) if titles else expA)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, legend_bbox_to_anchor_vertical), ncol=6)
            if suptitle:
                fig.suptitle(suptitle, fontsize=16)
            return fig, ax

        # Two-experiment case → side-by-side
        runsB = self.run_dictionary[expB]
        if labels is None:
            common = sorted(set(runsA.keys()) & set(runsB.keys()))
            if not common:
                raise ValueError(
                    f"No common labels between '{expA}' and '{expB}'. "
                    f"{expA} keys: {sorted(runsA.keys())}, {expB} keys: {sorted(runsB.keys())}"
                )
            use_labels = common
        else:
            use_labels = self._resolve_labels(labels)

        runs_orderedA = {k: runsA[k] for k in use_labels}
        runs_orderedB = {k: runsB[k] for k in use_labels}

        fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, sharey=sharey)
        plot_final_percentage_error_multi(
            runs=runs_orderedA,
            target=target,
            figsize=figsize,
            select_lowest_loss=select_lowest_loss,
            ax=ax[0],
        )
        plot_final_percentage_error_multi(
            runs=runs_orderedB,
            target=target,
            figsize=figsize,
            select_lowest_loss=select_lowest_loss,
            ax=ax[1],
        )
        ax[0].set_title(titles[0] if titles else expA)
        ax[1].set_title(titles[1] if titles else expB)
        ax[1].set_ylabel("")
        place_shared_legend(
            fig,
            ax.ravel(),
            empty_slot_only=False,
            bbox_to_anchor_vertical=legend_bbox_to_anchor_vertical,
        )
        return fig, ax

    def plot_tracked(
        self,
        experiments: Optional[Union[str, Tuple[str, str]]] = None,
        labels: Optional[Iterable[Union[int, str]]] = None,
        target=None,
        ax=None,
        color: Optional[str] = None,
        linestyles: Optional[Union[str, list[str]]] = None,
        figsize: Optional[Tuple[int, int]] = (11, 7),
        label_prefixes: Optional[Tuple[str, str]] = None,
        legend_title: Optional[str] = "Tracked Parameters:",
        legend_fontsize: Optional[int] = 12,
        legend_ncol: Optional[int] = None,
        legend_frameon: Optional[bool] = False,
        legend_bbox_to_anchor_vertical=-0.5,
        skip_elements: Tuple[str, ...] = ("callbacks",),
        **kwargs,
    ):
        """
        Plot tracked parameters for one or two experiments.

        experiments:
        - None: if two experiments loaded, overlay both; if one, plot that one.
        - "ExpA": just that experiment.
        - ("ExpA","ExpB"): overlay both.

        labels:
        - None: all labels in the chosen experiment (single), or the intersection (two).
        - Iterable: those labels (ints resolved via group_number_dict).
        """
        expA, expB = self._normalize_experiments(experiments)
        runsA = self.run_dictionary[expA]

        # Resolve labels default
        if expB is None:
            # single experiment
            use_labels = sorted(runsA.keys()) if labels is None else self._resolve_labels(labels)
        else:
            runsB = self.run_dictionary[expB]
            if labels is None:
                inter = sorted(set(runsA.keys()) & set(runsB.keys()))
                if not inter:
                    raise ValueError(f"No common labels between '{expA}' and '{expB}'.")
                use_labels = inter
            else:
                use_labels = self._resolve_labels(labels)

        # linestyles setup
        # check if linestyle is in kwargs
        if "linestyle" in kwargs:
            linestyles = kwargs["linestyle"]
            # remove default linestyle if present
            kwargs.pop("linestyle", None)

        if expB is None:
            lsA = (
                linestyles
                if isinstance(linestyles, str)
                else (linestyles[0] if isinstance(linestyles, list) else None)
            )
            prefixes = (f"{expA}_",)
        else:
            if linestyles is None:
                linestyles = ["-", "--"]
            if isinstance(linestyles, (list, tuple)):
                if len(linestyles) != 2:
                    raise ValueError(
                        "linestyles list must have length 2 when plotting two experiments."
                    )
                lsA, lsB = linestyles
            elif isinstance(linestyles, str):
                lsA = lsB = linestyles
            else:
                raise TypeError("linestyles must be a str or a list of two str.")

            prefixes = label_prefixes or (f"{expA}_", f"{expB}_")

        def _plot_one(runs: RunsDict, lbl: str, prefix: str, use_target, ls=None):
            if lbl not in runs:
                raise KeyError(f"Label '{lbl}' not found in experiment '{prefix}'.")
            return plot_tracked_parameters(
                history=runs[lbl],
                target=(target if use_target else None),
                label=f"{prefix}{lbl}",
                ax=ax,
                color=color,
                linestyle=ls,
                figsize=figsize,
                skip_elements=skip_elements,
                **kwargs,
            )

        first_fig_ax = None
        for idx, lbl in enumerate(use_labels):
            use_t = (idx == 0) and (target is not None)
            if expB is None:
                # single experiment: solid by default
                fa = _plot_one(runsA, lbl, prefixes[0], use_t, ls=lsA)
                if first_fig_ax is None and isinstance(fa, tuple):
                    first_fig_ax = fa
                    ax = fa[1]
            else:
                # two experiments overlay: A solid, B dashed
                fa = _plot_one(runsA, lbl, prefixes[0], use_t, ls=lsA)
                if first_fig_ax is None and isinstance(fa, tuple):
                    first_fig_ax = fa
                    ax = fa[1]
                _ = _plot_one(self.run_dictionary[expB], lbl, prefixes[1], False, ls=lsB)

        if first_fig_ax is not None:
            fig, ax = first_fig_ax
            place_shared_legend(
                fig,
                ax.ravel(),
                legend_title=legend_title,
                title_fontsize=legend_fontsize,
                frameon=legend_frameon,
                ncol=legend_ncol,
                bbox_to_anchor_vertical=legend_bbox_to_anchor_vertical,
            )

        return first_fig_ax if first_fig_ax is not None else (None, ax)

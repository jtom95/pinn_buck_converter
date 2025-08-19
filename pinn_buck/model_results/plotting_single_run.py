from typing import Iterable, List, Dict, Union, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from ..config import Parameters
from .history import TrainingHistory
import numpy as np
from matplotlib.transforms import blended_transform_factory as blend


def plot_tracked_parameters(
    history: TrainingHistory,
    figsize=(12, 8),
    skip_loss: bool = False,
    target: Parameters = None,
    logloss: bool = True,
    label: str = None,
    color: str = "black",
    ax: Iterable[plt.Axes] = None,
    optimizer_alpha: float = 0.06,  # shading strength
    optimizer_line_alpha: float = 0.4,
    skip_elements: Tuple[str, ...] = ("callbacks",),
    **kwargs,
):
    """
    Plot tracked physical parameters and (optionally) loss over entries.
    Ignores any 'epoch' resets; x-axis is simply the entry index.
    If 'optimizer' exists, shades segments and annotates optimizer names.
    """
    # Split the history into: extra columns (loss/optimizer/lr/epoch/etc.) and iterator (per-parameter)
    drop_condition = lambda c: (c in Parameters._fields) or ("Rload" in c)
    df_extra = history.df.drop(columns=[c for c in history.df.columns if drop_condition(c)])
    df_iter = history.df_from_iterator()  # per-parameter columns in display names

    # Decide which extra columns to plot
    has_opt = "optimizer" in df_extra.columns
    has_lr = "learning_rate" in df_extra.columns

    core_cols = list(df_extra.columns)
    for c in ("optimizer", "epoch") + skip_elements:  # we ignore epoch entirely
        if c in core_cols:
            core_cols.remove(c)
    if skip_loss and "loss" in core_cols:
        core_cols.remove("loss")

    # Join with iterator columns (keeps only rows common to both)
    df = df_extra[core_cols].join(df_iter, how="inner")

    # x-axis = simple entry index (0..N-1)
    N = len(df)
    x = np.arange(N, dtype=float)

    # Get optimizer labels aligned to df rows (if present)
    if has_opt:
        opt = history.df.loc[df.index, "optimizer"].to_numpy()
    else:
        opt = None

    # Build axes grid
    n = len(df.columns)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, sharex=True)
        axes = np.array(ax).reshape(nrows, ncols).flatten()
    else:
        ax = np.asarray(ax)
        if ax.shape != (nrows, ncols):
            raise ValueError(f"Expected axes shape {(nrows, ncols)}, got {ax.shape}")
        fig = ax[0, 0].figure
        axes = ax.flatten()

    # Targets
    target_dict = {name: value for name, value in target.iterator()} if target else {}

    # Helper to shade optimizer segments on a given axis
    def draw_optimizer_bands(ax_):
        if opt is None or len(opt) == 0:
            return
        starts = np.r_[0, 1 + np.flatnonzero(opt[1:] != opt[:-1])]
        ends = np.r_[starts[1:], len(opt)]
        names = [opt[s] for s in starts]

        palette = plt.cm.tab20.colors
        trans = blend(ax_.transData, ax_.transAxes)  # x in data; y in axes

        for k, (s, e) in enumerate(zip(starts, ends)):
            x0, x1 = float(s), float(e - 1 if e > s else s)
            if x1 == x0:
                x1 = x0 + 1.0  # minimal width for single-point segments

            # band under the curves
            ax_.axvspan(
                x0,
                x1,
                facecolor=palette[k % len(palette)],
                alpha=optimizer_alpha,
                linewidth=0.0,
                zorder=0,
            )

            # vertical separator at the start (except first)
            if s != 0:
                ax_.axvline(
                    x0,
                    color=palette[k % len(palette)],
                    alpha=optimizer_line_alpha,
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )

            # label inside the band (near top)
            xc = 0.5 * (x0 + x1)
            ax_.text(
                xc,
                0.92,
                names[k],
                rotation=90,
                va="top",
                ha="center",
                fontsize=9,
                color="k",
                alpha=0.85,
                transform=trans,
                zorder=2,
            )

    # Plot each series
    for i, col in enumerate(df.columns):
        ax_i = axes[i]
        y = df[col].to_numpy()

        if col == "loss":
            ax_i.plot(x, y, color=color, label=label or "loss", **kwargs)
            if logloss:
                ax_i.set_yscale("log")
                ax_i.set_ylabel("Loss (log scale)")
        else:
            if target and col in target_dict:
                ax_i.axhline(target_dict[col], color="black", lw=2.0, alpha=0.7, label="target")
            ax_i.plot(x, y, color=color, label=label or "estimate", **kwargs)

        ax_i.set_title(col)
        ax_i.set_xlabel("Entry")
        ax_i.grid(True, alpha=0.25)
        ax_i.legend(loc="best")

        draw_optimizer_bands(ax_i)

    # Learning rate panel, if there’s room
    last_used = len(df.columns)
    if has_lr and last_used < len(axes):
        ax_lr = axes[last_used]
        ax_lr.plot(
            x,
            history.df.loc[df.index, "learning_rate"].to_numpy(),
            color=color,
            label=label or "lr",
        )
        ax_lr.set_title("Learning Rate")
        ax_lr.set_xlabel("Entry")
        ax_lr.grid(True, alpha=0.25)
        ax_lr.legend(loc="best")
        draw_optimizer_bands(ax_lr)
        last_used += 1

    # Hide any empty subplots
    for j in range(last_used, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Tracked Parameters over Training", fontsize=16)
    return fig, ax


# plot the percentage error of the parameters
def plot_percentage_error_evolution(df: pd.DataFrame, target: Parameters):
    """
    Plot the percentage error of parameters compared to target values.

    Args:
        df (pd.DataFrame): DataFrame with parameter estimates.
        target (Parameters): Target parameters for reference.
    """
    if isinstance(df, TrainingHistory):
        df = df.df
    
    errors = {name: [] for name in Parameters._fields}

    for name in Parameters._fields:
        if name in df.columns:
            val = df[name].values
            nom = getattr(target, name)
            errors[name] = abs(val / nom - 1) * 100

    error_df = pd.DataFrame(errors)

    plt.figure(figsize=(12, 6))
    for name in Parameters._fields:
        if name in error_df.columns:
            plt.plot(error_df[name], label=name)

    plt.xlabel("Iteration (per 1000 steps)")
    plt.ylabel("Percentage Error (%)")
    plt.title("Percentage Error of Parameters")
    plt.legend()
    plt.grid(True)


def plot_final_percentage_error(
    training_run: TrainingHistory,
    target: Parameters,
    ax: plt.Axes = None,
    figsize=(6, 3),
    color="black",
    skip_params: List[str] = []
):
    """
    Plot the final percentage error of parameters compared to target values.

    Args:
        df (pd.DataFrame): DataFrame with parameter estimates.
        target (Parameters): Target parameters for reference.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    best_parameters: Parameters = training_run.best_parameters
    errors = {name: [] for name in Parameters._fields}
    ax.grid(True)
    for name in Parameters._fields:
        if name in training_run.df.columns and name not in skip_params:
            val = getattr(best_parameters, name)
            nom = getattr(target, name)
            errors[name] = abs(val / nom - 1) * 100
    # drop empty entries
    errors = {k: v for k, v in errors.items() if v}  # remove empty entries

    error_df = pd.DataFrame(errors, index=[0]).T
    error_df.plot(kind="bar", ax=ax, color=color, grid=True, legend=False)
    ax.set_ylabel("Percentage Error (%)")
    ax.set_title("Final Percentage Error of Parameters")
    plt.xticks(rotation=45)


def plot_final_percentage_error_multi(
    runs: Dict[str, "TrainingHistory"],  # {label → TrainingHistory}
    target: "Parameters",  # ground-truth parameters
    skip_params: Iterable[str] = (),
    ax: plt.Axes = None,
    figsize=(6, 3),
    palette: Iterable[str] = None,  # optional colour list
    select_lowest_loss: bool = True
):
    """
    Clustered bar-plot of |error| (%) for several TrainingHistorys.

    Parameters
    ----------
    runs : dict
        Keys   → legend labels
        Values → TrainingHistory instances (must expose .best_parameters & .df)
    target : Parameters
        Reference (ground-truth) parameter set.
    skip_params : iterable of str
        Parameter names to ignore.
    ax : plt.Axes, optional
        Existing axis to draw on.  If None, a new figure is created.
    figsize : tuple, optional
        Size used only when `ax` is None.
    palette : iterable of colour specs, optional
        One colour per run (cycled if shorter).
    """
    # 1. collect absolute percentage errors into a table
    rows: Dict[str, Dict[str, float]] = {}  # {param → {run_label → error}}

    for run_label, tr in runs.items():
        best_parameters = tr.get_best_parameters() if select_lowest_loss else tr.get_latest_parameters()  # get best parameters or last row
        for name, value in best_parameters.iterator():
            if name in tr.df_from_iterator() and name not in skip_params:
                target_value = target.get_from_iterator_name(name)
                err = abs(value / target_value - 1.0) * 100.0
                rows.setdefault(name, {})[run_label] = err

    # turn into (rows = parameters) × (cols = run labels) dataframe
    err_df = pd.DataFrame.from_dict(rows, orient="index").sort_index()

    # 2. plotting
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # pandas does the clustered layout automatically for kind="bar"
    err_df.plot(kind="bar", ax=ax, width=0.8, color=palette, grid=True)

    ax.set_ylabel("Percentage Error (%)")
    ax.set_title("Final Percentage Error of Parameters")
    ax.legend(title="Run")  # tweak ncol as you like
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return ax

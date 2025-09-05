from typing import List, Dict, Iterable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..parameters.parameter_class import Parameters
from ..training_history.history import TrainingHistory
from .io import Measurement


def inspect_repeated_lossy_data(
    lossy_meas: List[Measurement],
    label: str,
    reference: Measurement,
    slice_index: slice,
    ax=None,
    figsize=(12, 10),
    markersize=2,
    color="blue",
    plot_ideal: bool = True,
) -> List[plt.Axes]:
    """
    Inspect the repeated lossy data by plotting the measurements.
    """
    if plot_ideal:
        ax = reference.plot_data(
            label="ideal",
            sharex=True,
            ax=ax,
            slice_index=slice_index,
            legend=True,
            figsize=figsize,
            color="black",
            ignore_dt=True,
        )

    lossy_label = label
    for idx, meas in enumerate(lossy_meas):
        meas.plot_data(
            label=lossy_label,
            ax=ax,
            slice_index=slice_index,
            legend=True,
            markers=".",
            linestyle=" ",
            ignore_dt=True,
            markersize=markersize,
            color=color,
        )
        lossy_label = None
    return ax


def plot_repeated_tracked_parameters(
    runs: List[TrainingHistory],
    figsize=(12, 8),
    skip_loss: bool = False,
    target: Parameters = None,
    logloss: bool = True,
    label: str = None,
    color: str = "black",
    ax: Iterable[plt.Axes] = None,
):
    """
    Plot tracked physical parameters and optionally loss over iterations.

    Args:
        df (pd.DataFrame): DataFrame with loss and parameter columns.
        figsize (tuple): Figure size for the plot grid.
        skip_loss (bool): Whether to skip plotting the loss curve.
        target (Parameters, optional): If provided, plots horizontal reference lines for nominal values.
    """
    # create two new DataFrames, one with the mean and one with the std of the runs
    if not runs:
        raise ValueError("No runs provided for plotting.")
    # the means should be calculated over the repeated runs index by index
    df_grouping = pd.concat([tr.df for tr in runs], axis=0).groupby(level=0)
    df = df_grouping.mean().reset_index(drop=True)
    df_std = df_grouping.std().reset_index(drop=True)

    params = [col for col in df.columns if col != "loss"] if skip_loss else df.columns
    n = len(params)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        axes = ax.flatten()
    else:
        # check the axes have the correct number of subplots
        ax = np.asarray(ax)
        if ax.shape != (nrows, ncols):
            raise ValueError(f"Expected axes shape {(nrows, ncols)}, got {ax.shape}")

        fig = ax[0, 0].figure
        axes = ax.flatten()

    for i, param in enumerate(params):
        if param == "loss":
            (line,) = axes[i].plot(df[param], color=color, label=label)
            fill_color = line.get_color() if color is None else color
            axes[i].fill_between(
                df.index,
                df[param] - df_std[param],
                df[param] + df_std[param],
                color=fill_color,
                alpha=0.2,
                label=None,
            )
        else:
            (line,) = axes[i].plot(df[param], color=color, label=label if label else "estimate")
            fill_color = line.get_color() if color is None else color
            axes[i].fill_between(
                df.index,
                df[param] - df_std[param],
                df[param] + df_std[param],
                color=fill_color,
                alpha=0.2,
                label=None,
            )
            if target and hasattr(target, param):
                ref_val = getattr(target, param)
                axes[i].axhline(ref_val, color="red", linestyle="--", linewidth=1, label="target")
        axes[i].set_title(param)
        axes[i].set_xlabel("Iteration (per 1000 steps)")
        axes[i].grid(True)
        axes[i].legend()

    # if learning rate is in the DataFrame, plot it
    if "learning_rate" in df.columns:
        # on the last axis, plot the learning rate
        axes[-1].plot(df["learning_rate"], label=label, color=color)
        axes[-1].set_title("Learning Rate")
        axes[-1].set_xlabel("Iteration (per 1000 steps)")
        axes[-1].legend()
        i += 1  # avoid removing the last axis

    if logloss and "loss" in df.columns:
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Loss (log scale)")
        axes[0].legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Tracked Parameters over Training", fontsize=16)
    return fig, ax  # return only the used axes


def plot_final_percentage_error_multi_boxplot(
    runs: Dict[str, List["TrainingHistory"]],  # {label → List[TrainingHistory]}
    target: "Parameters",  # ground-truth parameters
    skip_params: Iterable[str] = (),
    ax: plt.Axes = None,
    figsize=(6, 3),
    palette: Iterable[str] = None,  # optional colour list
    select_lowest_loss: bool = True,
):
    """
    Clustered box-plot of |error| (%) for several TrainingHistorys.

    Parameters
    ----------
    runs : dict
        Keys   → legend labels
        Values → List of TrainingHistory instances (must expose .best_parameters & .df)
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
    rows: Dict[str, Dict[str, List[float]]] = {}  # {param → {run_label → [errors]}}

    for run_label, tr_list in runs.items():
        for tr in tr_list:
            bp = (
                tr.best_parameters if select_lowest_loss else tr.df.iloc[-1]
            )  # get best parameters or last row
            for name in Parameters._fields:
                if name in tr.df.columns and name not in skip_params:
                    err = abs(getattr(bp, name) / getattr(target, name) - 1.0) * 100.0
                    rows.setdefault(name, {}).setdefault(run_label, []).append(err)

    # turn into a long-form DataFrame for boxplot
    data = []
    for param, run_dict in rows.items():
        for run_label, errors in run_dict.items():
            for error in errors:
                data.append({"Parameter": param, "Run": run_label, "Error (%)": error})
    err_df = pd.DataFrame(data)

    # 2. plotting
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    sns.boxplot(data=err_df, x="Parameter", y="Error (%)", hue="Run", palette=palette, ax=ax)

    ax.set_ylabel("Percentage Error (%)")
    ax.set_title("Final Percentage Error of Parameters (Boxplot)")
    ax.legend(title="Run", ncol=min(len(runs), 4))  # tweak ncol as you like
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return ax

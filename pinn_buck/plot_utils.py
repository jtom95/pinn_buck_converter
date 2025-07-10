from typing import Iterable, List, Dict

import pandas as pd
import matplotlib.pyplot as plt
from .config import Parameters
from .io_model import TrainingRun
import numpy as np

def plot_tracked_parameters(
    df: pd.DataFrame,
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
    if isinstance(df, TrainingRun):
        df = df.df
    
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
            axes[i].plot(df[param], color=color, label=label)
        else: 
            axes[i].plot(df[param], color=color, label=label if label else "estimate")
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
        i += 1 # avoid removing the last axis

    if logloss and "loss" in df.columns:
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Loss (log scale)")
        axes[0].legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Tracked Parameters over Training", fontsize=16)
    return fig, ax  # return only the used axes


# plot the percentage error of the parameters
def plot_percentage_error_evolution(df: pd.DataFrame, target: Parameters):
    """
    Plot the percentage error of parameters compared to target values.

    Args:
        df (pd.DataFrame): DataFrame with parameter estimates.
        target (Parameters): Target parameters for reference.
    """
    if isinstance(df, TrainingRun):
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
    training_run: TrainingRun,
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
    runs: Dict[str, "TrainingRun"],  # {label → TrainingRun}
    target: "Parameters",  # ground-truth parameters
    skip_params: Iterable[str] = (),
    ax: plt.Axes = None,
    figsize=(6, 3),
    palette: Iterable[str] = None,  # optional colour list
    select_lowest_loss: bool = True
):
    """
    Clustered bar-plot of |error| (%) for several TrainingRuns.

    Parameters
    ----------
    runs : dict
        Keys   → legend labels
        Values → TrainingRun instances (must expose .best_parameters & .df)
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
        bp = tr.best_parameters if select_lowest_loss else tr.df.iloc[-1]  # get best parameters or last row
        for name in Parameters._fields:
            if name in tr.df.columns and name not in skip_params:
                err = abs(getattr(bp, name) / getattr(target, name) - 1.0) * 100.0
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
    ax.legend(title="Run", ncol=min(len(runs), 4))  # tweak ncol as you like
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return ax

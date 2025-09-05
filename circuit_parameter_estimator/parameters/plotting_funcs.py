from .._general_auxiliaries.plot_aux import place_shared_legend, _apply_eng_label_only
from .parameter_class import Parameters
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

def plot_parameter_priors(
    nominal: Parameters, 
    prior_sigma: Parameters, 
    true_params: Parameters, 
    ncols: int = 4, 
    figsize: Tuple[float, float] = (6, 6),
    legend_frameon: bool = True,
    legend_title: str = "Legend:",
    suptitle_fontsize: Optional[float | None] = 14,
    legend_title_fontsize: Optional[float | None] = 11
    ) -> tuple:
    # Plotting
    nrows = int(np.ceil(len(nominal) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, tight_layout=True)
    axes = axes.flatten()


    for ii, ((name, nominal), (_, sigma), (_, true_val)) in enumerate(
        zip(nominal.iterator(), prior_sigma.iterator(), true_params.iterator())
    ):

        # log-normal parameters
        mu = np.log(nominal)

        dist = lognorm(s=sigma, scale=np.exp(mu))

        x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 500)
        pdf = dist.pdf(x)

        ax = axes[ii]
        ax.plot(x, pdf, label=f"prior", color="black", linewidth=2)
        ax.axvline(true_val, color="red", linestyle="--", label="TRUE")
        ax.set_title(name)
        ax.set_yticks([])
        ax.legend()
        _apply_eng_label_only(ax, nominal.units.get_from_iterator_name(name))

    for ax in axes.flatten()[ii + 1 :]:
        ax.axis("off")
    place_shared_legend(
        fig, axes, empty_slot_only=True, frameon=legend_frameon, legend_title=legend_title, title_fontsize=legend_title_fontsize
    )


    fig.suptitle("Log-Normal Priors (linear space) with TRUE values", fontsize=suptitle_fontsize)
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    return fig, axes

from typing import Iterable, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def place_shared_legend(
    fig: Figure,
    axes: Iterable[Axes],
    *,
    ncol: int = None,
    empty_slot_only: bool = True,
    legend_title: str | None = None,
    frameon: bool = True,
    title_fontsize: int | None = None,
    bbox_to_anchor_horizontal: Optional[float] = 0.5,
    bbox_to_anchor_vertical: Optional[float] = -0.02
) -> None:
    """
    Consolidate a single legend:
      - If an empty subplot exists, put the legend there (centered) and hide that axis.
      - Else, put a flat legend under the entire figure.

    Removes any per-axes legends first.
    """
    axes = list(axes)

    # 3) find a "free" axis: no lines/collections/images
    def is_free(a: Axes) -> bool:
        return not (a.lines or a.collections or a.images)

    free_ax = next((a for a in axes if is_free(a)), None)
    
    if free_ax is None and empty_slot_only:
        # no free axis available
        return
    
    # 1) collect handles/labels for all unique labels across axes
    handles, labels = [], []
    seen_labels = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen_labels:
                seen_labels.add(label)
                handles.append(handle)
                labels.append(label)

    if not labels:
        return  # nothing to show

    # 2) clear existing legends on all axes
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if free_ax is not None:
        free_ax.axis("off")
        ncol = ncol or 1
        free_ax.legend(
            handles,
            labels,
            loc="center",
            frameon=frameon,
            ncol=ncol,
            title=legend_title,
            title_fontsize=title_fontsize,
        )
    else:
        # single, flat legend under the whole figure
        ncol = ncol or len(labels)  # if ncol is None, use number of labels
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(bbox_to_anchor_horizontal, bbox_to_anchor_vertical),
            ncol=ncol,
            frameon=frameon,
            title=legend_title,
            title_fontsize=title_fontsize,
        )
        # give a bit of room for the legend
        fig.subplots_adjust(bottom=0.15)

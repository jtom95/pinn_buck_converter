from typing import Iterable, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

_SI_PREFIX = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}


def _apply_eng_label_only(ax: plt.Axes, base_unit: str) -> None:
    """
    Auto-choose an SI prefix from current x-limits (multiple of 1e3),
    put it in the x-axis *label* like '[mV]', and show tick numbers in that scale.
    """

    def _update(_evt=None):
        # choose scale from current ticks (or xlims if ticks not ready)
        ticks = ax.get_xticks()
        if len(ticks) == 0 or not np.isfinite(ticks).any():
            x0, x1 = ax.get_xlim()
            magsrc = max(abs(x0), abs(x1))
        else:
            magsrc = max(abs(t) for t in ticks if np.isfinite(t)) or 0.0

        if magsrc <= 0:
            exp3 = 0
        else:
            exp3 = int(np.floor(np.log10(magsrc) / 3.0) * 3)
            exp3 = min(24, max(-24, exp3))  # clamp to supported prefixes

        scale = 10.0**exp3
        prefix = _SI_PREFIX.get(exp3, "")

        # formatter: numbers only, already divided by the chosen scale
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/scale:g}"))
        ax.set_xlabel(f"[{prefix}{base_unit}]")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        # ask for a redraw without infinite recursion
        ax.figure.canvas.draw_idle()

    # Update on draw and whenever x-lims change (zoom/pan)
    ax.callbacks.connect("xlim_changed", lambda a: _update())
    ax.figure.canvas.mpl_connect("draw_event", _update)

    _update()  # initial


def _layout_engine_kind(fig) -> str:
    """
    Return 'constrained' | 'tight' | 'none' | 'other' based on the engine's class name.
    Works with placeholder engines too.
    """
    eng = fig.get_layout_engine()
    if eng is None:
        return "none"
    cls = type(eng).__name__.lower()
    if "placeholderlayoutengine" in cls or "constrained" in cls:
        return "constrained"
    if "tight" in cls:
        return "tight"
    return "other"


def _reserve_bottom_space_for_legend(
    fig: plt.Figure,
    *,
    keep_constrained_layout: bool = False,
    extra_bottom_inch: float = 0.15,  # space to reserve under subplots
) -> None:
    """
    Ensure there's room under the subplots for a figure-level legend,
    using the *current* layout engine API (no deprecated calls).
    """
    eng = fig.get_layout_engine()
    kind = _layout_engine_kind(fig)

    if kind == "constrained":
        if keep_constrained_layout:
            # Prefer mutating the current engine if possible
            try:
                eng.set(h_pad=extra_bottom_inch*2)  # inches at top+bottom
            except Exception:
                # Or re-set the engine with desired pads
                fig.set_layout_engine("constrained", h_pad=extra_bottom_inch*2)
        else: 
            # change to tight engine
            fig.set_layout_engine("tight", rect=(0.0, max(0.0, extra_bottom_inch), 1.0, 1.0))
        return

    if kind == "tight":
        # Reserve a bottom band via rect; keep padding as default
        # rect=(left, bottom, right, top) in figure fraction coordinates
        try:
            # Try mutating the current engine
            eng.set(rect=(0.0, max(0.0, extra_bottom_inch), 1.0, 1.0))
        except Exception:
            # Or re-set tight with rect
            fig.set_layout_engine("tight", rect=(0.0, max(0.0, extra_bottom_inch), 1.0, 1.0))
        return

    # No engine (or an engine that *allows* subplots_adjust)
    if eng is None or getattr(eng, "adjust_compatible", True):
        fig.subplots_adjust(bottom=extra_bottom_inch)
        return

    # Engine present but incompatible → either disable or switch
    try:
        fig.set_layout_engine(None)  # then subplots_adjust works
        fig.subplots_adjust(bottom=extra_bottom_inch)
        return
    except Exception:
        # Last fallback: switch to tight with rect
        fig.set_layout_engine("tight", rect=(0.0, max(0.0, extra_bottom_inch), 1.0, 1.0))
        return


def place_shared_legend(
    fig: plt.Figure,
    axes: Iterable[plt.Axes],
    *,
    ncol: int | None = None,
    empty_slot_only: bool = True,
    legend_title: str | None = None,
    frameon: bool = True,
    title_fontsize: int | None = None,
    bbox_to_anchor_horizontal: float | None = 0.5,
    legend_bottom_inch: float = 0.15,
) -> None:
    """
    Consolidate a single legend:
      - If an empty subplot exists, put the legend there (centered) and hide that axis.
      - Else, put a flat legend under the entire figure.
    Adds bottom padding when constrained_layout=True so the legend fits.
    Removes any per-axes legends first.
    """
    axes = list(axes)

    def is_free(ax: plt.Axes) -> bool:
        return not any(
            [
                ax.has_data(),
                bool(ax.lines),
                bool(ax.collections),
                bool(ax.images),
                bool(ax.patches),
                bool(ax.containers),
                bool(ax.artists),
                any(t.get_text() for t in ax.texts),
            ]
        )

    free_ax = next((a for a in axes if is_free(a)), None)
    if free_ax is None and empty_slot_only:
        return

    # collect unique handles/labels
    handles, labels, seen = [], [], set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if not ll or ll in seen:
                continue
            seen.add(ll)
            handles.append(hh)
            labels.append(ll)
    if not labels:
        return

    # clear existing legends on all axes
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if free_ax is not None:
        free_ax.axis("off")
        (
            free_ax.legend(
                handles,
                labels,
                loc="center",
                frameon=frameon,
                ncol=(ncol or 1),
                title=legend_title,
                title_fontsize=title_fontsize,
            )
        )
        return

    # --- Figure-level legend path ---
    _reserve_bottom_space_for_legend(
        fig,
        extra_bottom_inch=legend_bottom_inch
    )

    # if engine_name == "constrained":
    #     # Add space in inches for constrained layout
    #     fig.set_constrained_layout_pads(h_pad=extra_bottom_pad_inch)
    # elif engine_name in (None, "none"):
    #     # No engine: subplots_adjust works
    #     fig.subplots_adjust(bottom=fallback_bottom_adjust)
    # else:
    #     # Some engine that blocks subplots_adjust → either disable or switch to tight w/rect
    #     try:
    #         fig.set_layout_engine(None)
    #         fig.subplots_adjust(bottom=fallback_bottom_adjust)
    #     except Exception:
    #         # fallback: use tight layout with a reserved rect
    #         fig.set_layout_engine("tight", rect=(0.0, max(0.0, fallback_bottom_adjust), 1.0, 1.0))

    # Place the legend centered under the figure
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(
            (0.5 if bbox_to_anchor_horizontal is None else bbox_to_anchor_horizontal),
            0.0,
        ),
        ncol=(ncol or len(labels)),
        frameon=frameon,
        title=legend_title,
        title_fontsize=title_fontsize,
    )
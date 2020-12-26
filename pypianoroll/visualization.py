"""Visualization tools.

Functions
---------

- plot_multitrack
- plot_pianoroll
- plot_track

"""
from typing import TYPE_CHECKING, List, Optional, Sequence

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from numpy import ndarray
from pretty_midi import (
    note_number_to_drum_name,
    note_number_to_name,
    program_to_instrument_class,
    program_to_instrument_name,
)

if TYPE_CHECKING:
    from .multitrack import Multitrack
    from .track import Track

__all__ = ["plot_multitrack", "plot_pianoroll", "plot_track"]


def plot_pianoroll(
    ax: Axes,
    pianoroll: ndarray,
    is_drum: bool = False,
    resolution: Optional[int] = None,
    downbeats: Optional[Sequence[int]] = None,
    preset: str = "full",
    cmap: str = "Blues",
    xtick: str = "auto",
    ytick: str = "octave",
    xticklabel: bool = True,
    yticklabel: str = "auto",
    tick_loc: Sequence[str] = ("bottom", "left"),
    tick_direction: str = "in",
    label: str = "both",
    grid_axis: str = "both",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
    **kwargs,
):
    """
    Plot a piano roll.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot the piano roll on.
    pianoroll : ndarray, shape=(?, 128), (?, 128, 3) or (?, 128, 4)
        Piano roll to plot. For a 3D piano-roll array, the last axis can
        be either RGB or RGBA.
    is_drum : bool
        Whether it is a percussion track. Defaults to False.
    resolution : int
        Time steps per quarter note. Required if `xtick` is 'beat'.
    downbeats : list
        Boolean array that indicates whether the time step contains a
        downbeat (i.e., the first time step of a bar).
    preset : {'full', 'frame', 'plain'}
        Preset theme. For 'full' preset, ticks, grid and labels are on.
        For 'frame' preset, ticks and grid are both off. For 'plain'
        preset, the x- and y-axis are both off. Defaults to 'full'.
    cmap : str or :class:`matplotlib.colors.Colormap`
        Colormap. Will be passed to :func:`matplotlib.pyplot.imshow`.
        Only effective when `pianoroll` is 2D. Defaults to 'Blues'.
    xtick : {'auto', 'beat', 'step', 'off'}
        Tick format for the x-axis. For 'auto' mode, set to 'beat' if
        `resolution` is given, otherwise set to 'step'. Defaults to
        'auto'.
    ytick : {'octave', 'pitch', 'off'}
        Tick format for the y-axis. Defaults to 'octave'.
    xticklabel : bool
        Whether to add tick labels along the x-axis.
    yticklabel : {'auto', 'name', 'number', 'off'}
        Tick label format for the y-axis. For 'name' mode, use pitch
        name as tick labels. For 'number' mode, use pitch number. For
        'auto' mode, set to 'name' if `ytick` is 'octave' and 'number'
        if `ytick` is 'pitch'. Defaults to 'auto'.
    tick_loc : sequence of {'bottom', 'top', 'left', 'right'}
        Tick locations. Defaults to `('bottom', 'left')`.
    tick_direction : {'in', 'out', 'inout'}
        Tick direction. Defaults to 'in'.
    label : {'x', 'y', 'both', 'off'}
        Whether to add labels to x- and y-axes. Defaults to 'both'.
    grid_axis : {'x', 'y', 'both', 'off'}
        Whether to add grids to the x- and y-axes. Defaults to 'both'.
    grid_linestyle : str
        Grid line style. Will be passed to
        :meth:`matplotlib.axes.Axes.grid`.
    grid_linewidth : float
        Grid line width. Will be passed to
        :meth:`matplotlib.axes.Axes.grid`.
    **kwargs
        Keyword arguments to be passed to
        :meth:`matplotlib.axes.Axes.imshow`.

    """
    # Plot the piano roll
    if pianoroll.ndim == 2:
        transposed = pianoroll.T
    elif pianoroll.ndim == 3:
        transposed = pianoroll.transpose(1, 0, 2)
    else:
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")

    img = ax.imshow(
        transposed,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1 if pianoroll.dtype == np.bool_ else 127,
        origin="lower",
        interpolation="none",
        **kwargs,
    )

    # Format ticks and labels
    if xtick == "auto":
        xtick = "beat" if resolution is not None else "step"
    elif xtick not in ("beat", "step", "off"):
        raise ValueError(
            "`xtick` must be one of 'auto', 'beat', 'step' or 'off', not "
            f"{xtick}."
        )
    if yticklabel == "auto":
        yticklabel = "name" if ytick == "octave" else "number"
    elif yticklabel not in ("name", "number", "off"):
        raise ValueError(
            "`yticklabel` must be one of 'auto', 'name', 'number' or 'off', "
            f"{yticklabel}."
        )

    if preset == "full":
        ax.tick_params(
            direction=tick_direction,
            bottom=("bottom" in tick_loc),
            top=("top" in tick_loc),
            left=("left" in tick_loc),
            right=("right" in tick_loc),
            labelbottom=xticklabel,
            labelleft=(yticklabel != "off"),
            labeltop=False,
            labelright=False,
        )
    elif preset == "frame":
        ax.tick_params(
            direction=tick_direction,
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
    elif preset == "plain":
        ax.axis("off")
    else:
        raise ValueError(
            f"`preset` must be one of 'full', 'frame' or 'plain', not {preset}"
        )

    # Format x-axis
    if xtick == "beat" and preset != "frame":
        if resolution is None:
            raise ValueError(
                "`resolution` must not be None when `xtick` is 'beat'."
            )
        n_beats = pianoroll.shape[0] // resolution
        ax.set_xticks(resolution * np.arange(n_beats) - 0.5)
        ax.set_xticklabels("")
        ax.set_xticks(
            resolution * (np.arange(n_beats) + 0.5) - 0.5, minor=True
        )
        ax.set_xticklabels(np.arange(1, n_beats + 1), minor=True)
        ax.tick_params(axis="x", which="minor", width=0)

    # Format y-axis
    if ytick == "octave":
        ax.set_yticks(np.arange(0, 128, 12))
        if yticklabel == "name":
            ax.set_yticklabels(["C{}".format(i - 2) for i in range(11)])
    elif ytick == "step":
        ax.set_yticks(np.arange(0, 128))
        if yticklabel == "name":
            if is_drum:
                ax.set_yticklabels(
                    [note_number_to_drum_name(i) for i in range(128)]
                )
            else:
                ax.set_yticklabels(
                    [note_number_to_name(i) for i in range(128)]
                )
    elif ytick != "off":
        raise ValueError(
            f"`ytick` must be one of 'octave', 'pitch' or 'off', not {ytick}."
        )

    # Format axis labels
    if label not in ("x", "y", "both", "off"):
        raise ValueError(
            f"`label` must be one of 'x', 'y', 'both' or 'off', not {label}."
        )

    if label in ("x", "both"):
        if xtick == "step" or not xticklabel:
            ax.set_xlabel("time (step)")
        else:
            ax.set_xlabel("time (beat)")

    if label in ("y", "both"):
        if is_drum:
            ax.set_ylabel("key name")
        else:
            ax.set_ylabel("pitch")

    # Plot the grid
    if grid_axis not in ("x", "y", "both", "off"):
        raise ValueError(
            "`grid` must be one of 'x', 'y', 'both' or 'off', not "
            f"{grid_axis}."
        )
    if grid_axis != "off":
        ax.grid(
            axis=grid_axis,
            color="k",
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
        )

    # Plot downbeat boundaries
    if downbeats is not None:
        for downbeat in downbeats:
            ax.axvline(x=downbeat, color="k", linewidth=1)

    return img


def plot_track(track: "Track", ax: Optional[Axes] = None, **kwargs) -> Axes:
    """
    Plot a track.

    Parameters
    ----------
    track : :class:`pypianoroll.Track`
        Track to plot.
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot the piano roll on. Defaults to call `plt.gca()`.
    **kwargs
        Keyword arguments to pass to :func:`pypianoroll.plot_pianoroll`.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        (Created) Axes object.

    """
    if ax is None:
        ax = plt.gca()
    plot_pianoroll(ax, track.pianoroll, track.is_drum, **kwargs)
    return ax


def _get_track_label(track_label, track=None):
    """Return corresponding track labels."""
    if track_label == "name":
        return track.name
    if track_label == "program":
        return program_to_instrument_name(track.program)
    if track_label == "family":
        return program_to_instrument_class(track.program)
    return track_label


def _add_tracklabel(ax, track_label, track=None):
    """Add a track label to an axis."""
    if not ax.get_ylabel():
        return
    ax.set_ylabel(
        f"{_get_track_label(track_label, track)}\n\n{ax.get_ylabel()}"
    )


def plot_multitrack(
    multitrack: "Multitrack",
    axs: Optional[Sequence[Axes]],
    mode: str = "separate",
    track_label: str = "name",
    preset: str = "full",
    cmaps: Optional[Sequence[str]] = None,
    xtick: str = "auto",
    ytick: str = "octave",
    xticklabel: bool = True,
    yticklabel: str = "auto",
    tick_loc: Sequence[str] = ("bottom", "left"),
    tick_direction: str = "in",
    label: str = "both",
    grid_axis: str = "both",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
    **kwargs,
) -> List[Axes]:
    """
    Plot the multitrack.

    Parameters
    ----------
    multitrack : :class:`pypianoroll.Multitrack`
        Multitrack to plot.
    axs : sequence of :class:`matplotlib.axes.Axes`
        Axes to plot the tracks on.
    mode : {'separate', 'blended', 'hybrid'}
        Plotting strategy for visualizing multiple tracks. For
        'separate' mode, plot each track separately. For 'blended',
        blend and plot the pianoroll as a colored image. For 'hybrid'
        mode, drum tracks are blended into a 'Drums' track and all
        other tracks are blended into an 'Others' track. Defaults to
        'separate'.
    track_label : {'name', 'program', 'family', 'off'}
        Track label format. When `mode` is 'hybrid', all options other
        than 'off' will label the two track with 'Drums' and 'Others'.
    preset : {'full', 'frame', 'plain'}
        Preset theme to use. For 'full' preset, ticks, grid and labels
        are on. For 'frame' preset, ticks and grid are both off. For
        'plain' preset, the x- and y-axis are both off. Defaults to
        'full'.
    cmaps :  tuple or list
        Colormaps. Will be passed to :func:`matplotlib.pyplot.imshow`.
        Only effective when `pianoroll` is 2D. Defaults to 'Blues'.
        If `mode` is 'separate', defaults to `('Blues', 'Oranges',
        'Greens', 'Reds', 'Purples', 'Greys')`. If `mode` is 'blended',
        defaults to `('hsv')`. If `mode` is 'hybrid', defaults to
        `('Blues', 'Greens')`.
    **kwargs
        Keyword arguments to pass to :func:`pypianoroll.plot_pianoroll`.

    Returns
    -------
    list of :class:`matplotlib.axes.Axes`
        (Created) list of Axes objects.

    """
    if not multitrack.tracks:
        raise RuntimeError("There is no track to plot.")
    if track_label not in ("name", "program", "family", "off"):
        raise ValueError(
            "`track_label` must be one of 'name', 'program' or 'family', not "
            f"{track_label}."
        )

    if axs is not None and not isinstance(axs, list):
        axs = list(axs)

    # Set default color maps
    if cmaps is None:
        if mode == "separate":
            cmaps = ("Blues", "Oranges", "Greens", "Reds", "Purples", "Greys")
        elif mode == "blended":
            cmaps = ("hsv",)
        else:
            cmaps = ("Blues", "Greens")

    n_tracks = len(multitrack.tracks)
    downbeats = multitrack.get_downbeat_steps()

    if mode == "separate":
        if axs is None:
            if n_tracks > 1:
                fig, axs_ = plt.subplots(n_tracks, sharex=True)
                fig.subplots_adjust(hspace=0)
                axs = axs_.tolist()
            else:
                fig, ax = plt.subplots()
                axs = [ax]

        for idx, track in enumerate(multitrack.tracks):
            now_xticklabel = xticklabel if idx < n_tracks else False
            plot_pianoroll(
                ax=axs[idx],
                pianoroll=track.pianoroll,
                is_drum=False,
                resolution=multitrack.resolution,
                downbeats=downbeats,
                preset=preset,
                cmap=cmaps[idx % len(cmaps)],
                xtick=xtick,
                ytick=ytick,
                xticklabel=now_xticklabel,
                yticklabel=yticklabel,
                tick_loc=tick_loc,
                tick_direction=tick_direction,
                label=label,
                grid_axis=grid_axis,
                grid_linestyle=grid_linestyle,
                grid_linewidth=grid_linewidth,
                **kwargs,
            )
            if track_label != "none":
                _add_tracklabel(axs[idx], track_label, track)

    elif mode == "blended":
        is_all_drum = True
        for track in multitrack.tracks:
            if not track.is_drum:
                is_all_drum = False

        if axs is None:
            fig, ax = plt.subplots()
            axs = [ax]

        stacked = multitrack.stack()

        colormap = matplotlib.cm.get_cmap(cmaps[0])
        colormatrix = colormap(np.arange(0, 1, 1 / n_tracks))[:, :3]
        recolored = np.clip(
            np.matmul(stacked.reshape(-1, n_tracks), colormatrix), 0, 1
        )
        blended = recolored.reshape(stacked.shape[1:] + (3,))

        plot_pianoroll(
            ax=axs[0],
            pianoroll=blended,
            is_drum=is_all_drum,
            resolution=multitrack.resolution,
            downbeats=downbeats,
            preset=preset,
            xtick=xtick,
            ytick=ytick,
            xticklabel=xticklabel,
            yticklabel=yticklabel,
            tick_loc=tick_loc,
            tick_direction=tick_direction,
            label=label,
            grid_axis=grid_axis,
            grid_linestyle=grid_linestyle,
            grid_linewidth=grid_linewidth,
            **kwargs,
        )

        if track_label != "none":
            patches = [
                Patch(
                    color=colormatrix[idx],
                    label=_get_track_label(track_label, track),
                )
                for idx, track in enumerate(multitrack.tracks)
            ]
            plt.legend(handles=patches)

    elif mode == "hybrid":
        drums = multitrack.copy()
        drums.tracks = [track for track in multitrack.tracks if track.is_drum]
        merged_drums = drums.blend()

        others = multitrack.copy()
        others.tracks = [
            track for track in multitrack.tracks if not track.is_drum
        ]
        merged_others = others.blend()

        if axs is None:
            fig, axs_ = plt.subplots(2, sharex=True, sharey=True)
            axs = axs_.tolist()

        plot_pianoroll(
            axs[0],
            merged_drums,
            True,
            multitrack.resolution,
            downbeats,
            preset=preset,
            cmap=cmaps[0],
            xtick=xtick,
            ytick=ytick,
            xticklabel=xticklabel,
            yticklabel=yticklabel,
            tick_loc=tick_loc,
            tick_direction=tick_direction,
            label=label,
            grid_axis=grid_axis,
            grid_linestyle=grid_linestyle,
            grid_linewidth=grid_linewidth,
            **kwargs,
        )
        plot_pianoroll(
            axs[1],
            merged_others,
            False,
            multitrack.resolution,
            downbeats,
            preset=preset,
            cmap=cmaps[1],
            ytick=ytick,
            xticklabel=xticklabel,
            yticklabel=yticklabel,
            tick_loc=tick_loc,
            tick_direction=tick_direction,
            label=label,
            grid_axis=grid_axis,
            grid_linestyle=grid_linestyle,
            grid_linewidth=grid_linewidth,
            **kwargs,
        )
        fig.subplots_adjust(hspace=0)

        if track_label != "none":
            _add_tracklabel(axs[0], "Drums")
            _add_tracklabel(axs[1], "Others")

    else:
        raise ValueError(
            "`mode` must be one of 'separate', 'blended' or 'hybrid', not"
            f"{mode}."
        )

    return axs  # type: ignore

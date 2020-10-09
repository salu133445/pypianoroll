"""Module for visualizing multitrack piano rolls."""
from typing import Optional, Sequence

import matplotlib
import numpy as np
import pretty_midi
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from numpy import ndarray

try:
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

__all__ = ["plot_multitrack", "plot_pianoroll", "plot_track", "save_animation"]


def plot_pianoroll(
    ax,
    pianoroll: ndarray,
    is_drum: bool = False,
    resolution: Optional[int] = None,
    downbeats: Optional[Sequence[int]] = None,
    preset: str = "default",
    cmap: str = "Blues",
    xtick: str = "auto",
    ytick: str = "octave",
    xticklabel: bool = True,
    yticklabel: str = "auto",
    tick_loc: Optional[Sequence[str]] = None,
    tick_direction: str = "in",
    label: str = "both",
    grid: str = "both",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
):
    """
    Plot a piano roll given as a numpy array.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object where the piano roll will be plotted on.
    pianoroll : np.ndarray
        A piano roll to be plotted. The values should be in [0, 1] when
        data type is float, and in [0, 127] when data type is integer.

        - For a 2D array, shape=(num_time_step, num_pitch).
        - For a 3D array, shape=(num_time_step, num_pitch, num_channel),
          where channels can be either RGB or RGBA.

    is_drum : bool
        A boolean number that indicates whether it is a percussion
        track. Defaults to False.
    resolution : int
        Number of time steps used to represent a beat. Required and only
        effective when `xtick` is 'beat'.
    downbeats : list
        An array that indicates whether the time step contains a
        downbeat (i.e., the first time step of a bar).

    preset : {'default', 'plain', 'frame'}
        A string that indicates the preset theme to use.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        The colormap to use in :func:`matplotlib.pyplot.imshow`.
        Defaults to 'Blues'. Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        A string that indicates what to use as ticks along the x-axis.
        If 'auto' is given, automatically set to 'beat' if `resolution`
        is also given and set to 'step', otherwise. Defaults to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        A string that indicates what to use as ticks along the y-axis.
        Defaults to 'octave'.
    xticklabel : bool
        Whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when
        `is_drum` is True) as tick labels along the y-axis. If 'number',
        use pitch number. If 'auto', set to 'name' when `ytick` is
        'octave' and 'number' when `ytick` is 'pitch'. Defaults to
        'auto'. Only effective when `ytick` is not 'off'.
    tick_loc : tuple or list
        The locations to put the ticks. Availables elements are
        'bottom', 'top', 'left' and 'right'. Defaults to ('bottom',
        'left').
    tick_direction : {'in', 'out', 'inout'}
        A string that indicates where to put the ticks. Defaults to
        'in'. Only effective when one of `xtick` and `ytick` is on.
    label : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add labels to the x-axis and
        y-axis. Defaults to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add grids to the x-axis,
        y-axis, both or neither. Defaults to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linestyle' argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linewidth' argument.

    """
    if pianoroll.ndim not in (2, 3):
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")
    if pianoroll.shape[1] != 128:
        raise ValueError(
            "The length of the second axis of `pianoroll` must be 128."
        )
    if xtick not in ("auto", "beat", "step", "off"):
        raise ValueError(
            "`xtick` must be one of {'auto', 'beat', 'step', 'none'}."
        )
    if xtick == "beat" and resolution is None:
        raise ValueError(
            "`resolution` must be specified when `xtick` is 'beat'."
        )
    if ytick not in ("octave", "pitch", "off"):
        raise ValueError("`ytick` must be one of {octave', 'pitch', 'off'}.")
    if not isinstance(xticklabel, bool):
        raise TypeError("`xticklabel` must be bool.")
    if yticklabel not in ("auto", "name", "number", "off"):
        raise ValueError(
            "`yticklabel` must be one of {'auto', 'name', 'number', 'off'}."
        )
    if tick_direction not in ("in", "out", "inout"):
        raise ValueError(
            "`tick_direction` must be one of {'in', 'out', 'inout'}."
        )
    if label not in ("x", "y", "both", "off"):
        raise ValueError("`label` must be one of {'x', 'y', 'both', 'off'}.")
    if grid not in ("x", "y", "both", "off"):
        raise ValueError("`grid` must be one of {'x', 'y', 'both', 'off'}.")

    # plotting
    if pianoroll.ndim > 2:
        to_plot = pianoroll.transpose(1, 0, 2)
    else:
        to_plot = pianoroll.T
    if np.issubdtype(pianoroll.dtype, np.bool_) or np.issubdtype(
        pianoroll.dtype, np.floating
    ):
        ax.imshow(
            to_plot,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="none",
        )
    elif np.issubdtype(pianoroll.dtype, np.integer):
        ax.imshow(
            to_plot,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=127,
            origin="lower",
            interpolation="none",
        )
    else:
        raise TypeError("Unsupported data type for `pianoroll`.")

    # tick setting
    if tick_loc is None:
        tick_loc = ("bottom", "left")
    if xtick == "auto":
        xtick = "beat" if resolution is not None else "step"
    if yticklabel == "auto":
        yticklabel = "name" if ytick == "octave" else "number"

    if preset == "plain":
        ax.axis("off")
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
    else:
        ax.tick_params(
            direction=tick_direction,
            bottom=("bottom" in tick_loc),
            top=("top" in tick_loc),
            left=("left" in tick_loc),
            right=("right" in tick_loc),
            labelbottom=(xticklabel != "off"),
            labelleft=(yticklabel != "off"),
            labeltop=False,
            labelright=False,
        )

    # x-axis
    if xtick == "beat" and preset != "frame":
        num_beat = pianoroll.shape[0] // resolution
        ax.set_xticks(resolution * np.arange(num_beat) - 0.5)
        ax.set_xticklabels("")
        ax.set_xticks(
            resolution * (np.arange(num_beat) + 0.5) - 0.5, minor=True
        )
        ax.set_xticklabels(np.arange(1, num_beat + 1), minor=True)
        ax.tick_params(axis="x", which="minor", width=0)

    # y-axis
    if ytick == "octave":
        ax.set_yticks(np.arange(0, 128, 12))
        if yticklabel == "name":
            ax.set_yticklabels(["C{}".format(i - 2) for i in range(11)])
    elif ytick == "step":
        ax.set_yticks(np.arange(0, 128))
        if yticklabel == "name":
            if is_drum:
                ax.set_yticklabels(
                    [
                        pretty_midi.note_number_to_drum_name(i)
                        for i in range(128)
                    ]
                )
            else:
                ax.set_yticklabels(
                    [pretty_midi.note_number_to_name(i) for i in range(128)]
                )

    # axis labels
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

    # grid
    if grid != "off":
        ax.grid(
            axis=grid,
            color="k",
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
        )

    # downbeat boarder
    if downbeats is not None and preset != "plain":
        for step in downbeats:
            ax.axvline(x=step, color="k", linewidth=1)


def plot_track(
    track,
    filename: Optional[str] = None,
    resolution: Optional[int] = None,
    downbeats: Optional[Sequence[int]] = None,
    preset: str = "default",
    cmap: str = "Blues",
    xtick: str = "auto",
    ytick: str = "octave",
    xticklabel=True,
    yticklabel: str = "auto",
    tick_loc: Optional[Sequence[str]] = None,
    tick_direction: str = "in",
    label: str = "both",
    grid: str = "both",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
):
    """
    Plot the pianoroll or save a plot of the pianoroll.

    Parameters
    ----------
    filename :
        Filename to which the plot is saved. If None, save nothing.
    resolution : int
        Number of time steps used to represent a beat. Required and only
        effective when `xtick` is 'beat'.
    downbeats : list
        An array that indicates whether the time step contains a
        downbeat (i.e., the first time step of a bar).

    preset : {'default', 'plain', 'frame'}
        A string that indicates the preset theme to use.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        Colormap to use in :func:`matplotlib.pyplot.imshow`. Defaults
        to 'Blues'. Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        A string that indicates what to use as ticks along the x-axis.
        If 'auto' is given, automatically set to 'beat' if `resolution`
        is also given and set to 'step', otherwise. Defaults to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        A string that indicates what to use as ticks along the y-axis.
        Defaults to 'octave'.
    xticklabel : bool
        Whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when
        `is_drum` is True) as tick labels along the y-axis. If 'number',
        use pitch number. If 'auto', set to 'name' when `ytick` is
        'octave' and 'number' when `ytick` is 'pitch'. Defaults to
        'auto'. Only effective when `ytick` is not 'off'.
    tick_loc : tuple or list
        Locations to put the ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. Defaults to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        A string that indicates where to put the ticks. Defaults to
        'in'. Only effective when one of `xtick` and `ytick` is on.
    label : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add labels to the x-axis and
        y-axis. Defaults to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add grids to the x-axis,
        y-axis, both or neither. Defaults to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linestyle' argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linewidth' argument.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        Created Figure object.
    :class:`matplotlib.axes.Axes`
        Created Axes object.

    """
    fig, ax = plt.subplots()
    plot_pianoroll(
        ax,
        track.pianoroll,
        track.is_drum,
        resolution,
        downbeats,
        preset=preset,
        cmap=cmap,
        xtick=xtick,
        ytick=ytick,
        xticklabel=xticklabel,
        yticklabel=yticklabel,
        tick_loc=tick_loc,
        tick_direction=tick_direction,
        label=label,
        grid=grid,
        grid_linestyle=grid_linestyle,
        grid_linewidth=grid_linewidth,
    )

    if filename is not None:
        plt.savefig(filename)

    return fig, ax


def plot_multitrack(
    multitrack,
    filename: Optional[str] = None,
    mode: str = "separate",
    track_label: str = "name",
    preset: str = "default",
    cmaps: Optional[Sequence[str]] = None,
    xtick: str = "auto",
    ytick: str = "octave",
    xticklabel: bool = True,
    yticklabel: str = "auto",
    tick_loc: Optional[Sequence[str]] = None,
    tick_direction: str = "in",
    label: str = "both",
    grid: str = "both",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
):
    """
    Plot the pianorolls or save a plot of them.

    Parameters
    ----------
    filename : str
        Filename to which the plot is saved. If None, save nothing.
    mode : {'separate', 'stacked', 'hybrid'}
        A string that indicate the plotting mode to use. Defaults to
        'separate'.

        - In 'separate' mode, all the tracks are plotted separately.
        - In 'stacked' mode, a color is assigned based on `cmaps` to
          the piano roll of each track and the piano rolls are stacked
          and plotted as a colored image with RGB channels.
        - In 'hybrid' mode, the drum tracks are merged into a 'Drums'
          track, while the other tracks are merged into an 'Others'
          track, and the two merged tracks are then plotted separately.

    track_label : {'name', 'program', 'family', 'off'}
        A sting that indicates what to use as labels to the track. When
        `mode` is 'hybrid', all options other than 'off' will label the
        two track with 'Drums' and 'Others'.
    preset : {'default', 'plain', 'frame'}
        A string that indicates the preset theme to use.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmaps :  tuple or list
        The `matplotlib.colors.Colormap` instances or colormap codes to
        use.

        - When `mode` is 'separate', each element will be passed to each
          call of :func:`matplotlib.pyplot.imshow`. Defaults to
          ('Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Greys').
        - When `mode` is stacked, a color is assigned based on `cmaps`
          to the piano roll of each track. Defaults to ('hsv').
        - When `mode` is 'hybrid', the first (second) element is used
          in the 'Drums' ('Others') track. Defaults to ('Blues',
          'Greens').

    xtick : {'auto', 'beat', 'step', 'off'}
        A string that indicates what to use as ticks along the x-axis.
        If 'auto' is given, automatically set to 'beat' if `resolution`
        is also given and set to 'step', otherwise. Defaults to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        A string that indicates what to use as ticks along the y-axis.
        Defaults to 'octave'.
    xticklabel : bool
        Whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when
        `is_drum` is True) as tick labels along the y-axis. If 'number',
        use pitch number. If 'auto', set to 'name' when `ytick` is
        'octave' and 'number' when `ytick` is 'pitch'. Defaults to
        'auto'. Only effective when `ytick` is not 'off'.
    tick_loc : tuple or list
        Locations to put the ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. Defaults to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        A string that indicates where to put the ticks. Defaults to
        'in'. Only effective when one of `xtick` and `ytick` is on.
    label : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add labels to the x-axis and
        y-axis. Defaults to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add grids to the x-axis,
        y-axis, both or neither. Defaults to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linestyle' argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linewidth' argument.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        Created Figure object.
    list of :class:`matplotlib.axes.Axes`
        Created list of Axes objects.

    """

    def get_track_label(track_label, track=None):
        """Return corresponding track labels."""
        if track_label == "name":
            return track.name
        if track_label == "program":
            return pretty_midi.program_to_instrument_name(track.program)
        if track_label == "family":
            return pretty_midi.program_to_instrument_class(track.program)
        return track_label

    def add_tracklabel(ax, track_label, track=None):
        """Add a track label to an axis."""
        if not ax.get_ylabel():
            return
        ax.set_ylabel(
            get_track_label(track_label, track) + "\n\n" + ax.get_ylabel()
        )

    multitrack.validate()
    if not multitrack.tracks:
        raise ValueError("There is no track to plot.")
    if mode not in ("separate", "stacked", "hybrid"):
        raise ValueError(
            "`mode` must be one of {'separate', 'stacked', 'hybrid'}."
        )
    if track_label not in ("name", "program", "family", "off"):
        raise ValueError(
            "`track_label` must be one of {'name', 'program', 'family'}."
        )

    if cmaps is None:
        if mode == "separate":
            cmaps = ("Blues", "Oranges", "Greens", "Reds", "Purples", "Greys")
        elif mode == "stacked":
            cmaps = ("hsv",)
        else:
            cmaps = ("Blues", "Greens")

    num_track = len(multitrack.tracks)
    downbeats = multitrack.get_downbeat_steps()

    if mode == "separate":
        if num_track > 1:
            fig, axs = plt.subplots(num_track, sharex=True)
        else:
            fig, ax = plt.subplots()
            axs = [ax]

        for idx, track in enumerate(multitrack.tracks):
            now_xticklabel = xticklabel if idx < num_track else False
            plot_pianoroll(
                axs[idx],
                track.pianoroll,
                False,
                multitrack.resolution,
                downbeats,
                preset=preset,
                cmap=cmaps[idx % len(cmaps)],
                xtick=xtick,
                ytick=ytick,
                xticklabel=now_xticklabel,
                yticklabel=yticklabel,
                tick_loc=tick_loc,
                tick_direction=tick_direction,
                label=label,
                grid=grid,
                grid_linestyle=grid_linestyle,
                grid_linewidth=grid_linewidth,
            )
            if track_label != "none":
                add_tracklabel(axs[idx], track_label, track)

        if num_track > 1:
            fig.subplots_adjust(hspace=0)

        if filename is not None:
            plt.savefig(filename)

        return (fig, axs)

    if mode == "stacked":
        is_all_drum = True
        for track in multitrack.tracks:
            if not track.is_drum:
                is_all_drum = False

        fig, ax = plt.subplots()
        stacked = multitrack.get_stacked_pianoroll()

        colormap = matplotlib.cm.get_cmap(cmaps[0])
        colormatrix = colormap(np.arange(0, 1, 1 / num_track))[:, :3]
        recolored = np.clip(
            np.matmul(stacked.reshape(-1, num_track), colormatrix), 0, 1
        )
        stacked = recolored.reshape(stacked.shape[:2] + (3,))

        plot_pianoroll(
            ax,
            stacked,
            is_all_drum,
            multitrack.resolution,
            downbeats,
            preset=preset,
            xtick=xtick,
            ytick=ytick,
            xticklabel=xticklabel,
            yticklabel=yticklabel,
            tick_loc=tick_loc,
            tick_direction=tick_direction,
            label=label,
            grid=grid,
            grid_linestyle=grid_linestyle,
            grid_linewidth=grid_linewidth,
        )

        if track_label != "none":
            patches = [
                Patch(
                    color=colormatrix[idx],
                    label=get_track_label(track_label, track),
                )
                for idx, track in enumerate(multitrack.tracks)
            ]
            plt.legend(handles=patches)

        if filename is not None:
            plt.savefig(filename)

        return (fig, [ax])

    if mode == "hybrid":
        drums = [
            i for i, track in enumerate(multitrack.tracks) if track.is_drum
        ]
        others = [i for i in range(len(multitrack.tracks)) if i not in drums]
        merged_drums = multitrack.get_merged_pianoroll(drums)
        merged_others = multitrack.get_merged_pianoroll(others)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        plot_pianoroll(
            ax1,
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
            grid=grid,
            grid_linestyle=grid_linestyle,
            grid_linewidth=grid_linewidth,
        )
        plot_pianoroll(
            ax2,
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
            grid=grid,
            grid_linestyle=grid_linestyle,
            grid_linewidth=grid_linewidth,
        )
        fig.subplots_adjust(hspace=0)

        if track_label != "none":
            add_tracklabel(ax1, "Drums")
            add_tracklabel(ax2, "Others")

        if filename is not None:
            plt.savefig(filename)

        return (fig, [ax1, ax2])


def save_animation(
    filename: str,
    pianoroll: ndarray,
    fps: int,
    window: int,
    hop: int = 1,
    is_drum: bool = False,
    resolution: Optional[int] = None,
    downbeats: Optional[Sequence[int]] = None,
    preset: str = "default",
    cmap: str = "Blues",
    xtick: str = "auto",
    ytick: str = "octave",
    xticklabel: bool = True,
    yticklabel: str = "auto",
    tick_loc: Optional[Sequence[str]] = None,
    tick_direction: str = "in",
    label: str = "both",
    grid: str = "both",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
    **kwargs
):
    """
    Save a piano roll to an animation in video or GIF format.

    Parameters
    ----------
    filename : str
        Filename to which the animation is saved.
    pianoroll : np.ndarray
        A piano roll to be plotted. The values should be in [0, 1] when
        data type is float, and in [0, 127] when data type is integer.

        - For a 2D array, shape=(num_time_step, num_pitch).
        - For a 3D array, shape=(num_time_step, num_pitch, num_channel),
          where channels can be either RGB or RGBA.

    fps : int
        Number of frames per second in the resulting video or GIF file.
    window : int
        Window size to be applied to `pianoroll` for the animation.
    hop : int
        Hop size to be applied to `pianoroll` for the animation.
    is_drum : bool
        Whether it is a percussion track. Defaults to False.
    resolution : int
        Number of time steps used to represent a beat. Required and only
        effective when `xtick` is 'beat'.
    downbeats : list
        An array that indicates whether the time step contains a
        downbeat (i.e., the first time step of a bar).

    preset : {'default', 'plain', 'frame'}
        A string that indicates the preset theme to use.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        Colormap to use in :func:`matplotlib.pyplot.imshow`. Defaults
        to 'Blues'. Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        A string that indicates what to use as ticks along the x-axis.
        If 'auto' is given, automatically set to 'beat' if `resolution`
        is also given and set to 'step', otherwise. Defaults to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        A string that indicates what to use as ticks along the y-axis.
        Defaults to 'octave'.
    xticklabel : bool
        Whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when
        `is_drum` is True) as tick labels along the y-axis. If 'number',
        use pitch number. If 'auto', set to 'name' when `ytick` is
        'octave' and 'number' when `ytick` is 'pitch'. Defaults to
        'auto'. Only effective when `ytick` is not 'off'.
    tick_loc : tuple or list
        Locations to put the ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. Defaults to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        A string that indicates where to put the ticks. Defaults to
        'in'. Only effective when one of `xtick` and `ytick` is on.
    label : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add labels to the x-axis and
        y-axis. Defaults to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add grids to the x-axis,
        y-axis, both or neither. Defaults to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linestyle' argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as
        'linewidth' argument.

    """
    if not HAS_MOVIEPY:
        raise ImportError(
            "moviepy package is required for animation supports."
        )

    def make_frame(t):
        """Return an image of the frame for time t."""
        fig = plt.gcf()
        ax = plt.gca()
        f_idx = int(t * fps)
        start = hop * f_idx
        end = start + window
        to_plot = transposed[:, start:end]
        extent = (start, end - 1, 0, 127)
        ax.imshow(
            to_plot,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            interpolation="none",
            extent=extent,
        )

        if xtick == "beat":
            next_major_idx = resolution - start % resolution
            if start % resolution < resolution // 2:
                next_minor_idx = resolution // 2 - start % resolution
            else:
                next_minor_idx = (
                    resolution // 2 - start % resolution + resolution
                )
            xticks_major = np.arange(next_major_idx, window, resolution)
            xticks_minor = np.arange(next_minor_idx, window, resolution)
            if end % resolution < resolution // 2:
                last_minor_idx = resolution // 2 - end % resolution
            else:
                last_minor_idx = (
                    resolution // 2 - end % resolution + resolution
                )
            xtick_labels = np.arange(
                (start + next_minor_idx) // resolution,
                (end + last_minor_idx) // resolution,
            )
            ax.set_xticks(xticks_major)
            ax.set_xticklabels("")
            ax.set_xticks(xticks_minor, minor=True)
            ax.set_xticklabels(xtick_labels, minor=True)
            ax.tick_params(axis="x", which="minor", width=0)

        return mplfig_to_npimage(fig)

    if xtick == "auto":
        xtick = "beat" if resolution is not None else "step"

    _, ax = plt.subplots()
    plot_pianoroll(
        ax,
        pianoroll[:window],
        is_drum,
        resolution,
        downbeats,
        preset=preset,
        cmap=cmap,
        xtick=xtick,
        ytick=ytick,
        xticklabel=xticklabel,
        yticklabel=yticklabel,
        tick_loc=tick_loc,
        tick_direction=tick_direction,
        label=label,
        grid=grid,
        grid_linestyle=grid_linestyle,
        grid_linewidth=grid_linewidth,
    )

    num_frame = int((pianoroll.shape[0] - window) / hop)
    duration = int(num_frame / fps)

    if np.issubdtype(pianoroll.dtype, np.bool_) or np.issubdtype(
        pianoroll.dtype, np.floating
    ):
        vmax = 1
    elif np.issubdtype(pianoroll.dtype, np.integer):
        vmax = 127
    else:
        raise TypeError("Unsupported data type for `pianoroll`.")
    vmin = 0

    transposed = pianoroll.T
    animation = VideoClip(make_frame, duration=duration)
    if filename.endswith(".gif"):
        animation.write_gif(filename, fps, **kwargs)
    else:
        animation.write_videofile(filename, fps, **kwargs)
    plt.close()

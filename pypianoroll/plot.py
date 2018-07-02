"""Module for plotting multi-track and single-track piano-rolls.

"""
from __future__ import absolute_import, division, print_function
import numpy as np
import pretty_midi

try:
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

def plot_pianoroll(ax, pianoroll, is_drum=False, beat_resolution=None,
                   downbeats=None, preset='default', cmap='Blues', xtick='auto',
                   ytick='octave', xticklabel=True, yticklabel='auto',
                   tick_loc=None, tick_direction='in', label='both',
                   grid='both', grid_linestyle=':', grid_linewidth=.5):
    """
    Plot a piano-roll given as a numpy array.

    Parameters
    ----------
    ax : matplotlib.axes.Axes object
         The :class:`matplotlib.axes.Axes` object where the piano-roll will
         be plotted on.
    pianoroll : np.ndarray
        The piano-roll to be plotted. The values should be in [0, 1] when data
        type is float, and in [0, 127] when data type is integer.

        - For a 2D array, shape=(num_time_step, num_pitch).
        - For a 3D array, shape=(num_time_step, num_pitch, num_channel),
          where channels can be either RGB or RGBA.

    is_drum : bool
        Drum indicator. True for drums. False for other instruments. Default
        to False.
    beat_resolution : int
        Resolution of a beat (in time step). Required and only effective
        when `xticklabel` is 'beat'.
    downbeats : list
        Indices of time steps that contain downbeats., i.e. the first time
        step of a bar.
    preset : {'default', 'plain', 'frame'}
        Preset themes for the plot.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        Colormap to use in :func:`matplotlib.pyplot.imshow`. Default to
        'Blues'. Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        Use beat number or step number as ticks along the x-axis, or
        automatically set to 'beat' when `beat_resolution` is given and set
        to 'step', otherwise. Default to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        Use octave or pitch as ticks along the y-axis. Default to 'octave'.
    xticklabel : bool
        Indicate whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum`
        is True) as tick labels along the y-axis. If 'number', use pitch
        number. If 'auto', set to 'name' when `ytick` is 'octave' and
        'number' when `ytick` is 'pitch'. Default to 'auto'. Only effective
        when `ytick` is not 'off'.
    tick_loc : tuple or list
        List of locations to put ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. If None, default to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        Put ticks inside the axes, outside the axes, or both. Default to
        'in'. Only effective when `xtick` and `ytick` are not both 'off'.
    label : {'x', 'y', 'both', 'off'}
        Add label to the x-axis, y-axis, both or neither. Default to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        Add grid to the x-axis, y-axis, both or neither. Default to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linestyle'
        argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linewidth'
        argument.

    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib package is required for plotting "
                          "supports.")

    if pianoroll.ndim not in (2, 3):
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")
    if pianoroll.shape[1] != 128:
        raise ValueError("The shape of `pianoroll` must be (num_time_step, "
                         "128)")
    if xtick not in ('auto', 'beat', 'step', 'off'):
        raise ValueError("`xtick` must be one of {'auto', 'beat', 'step', "
                         "'none'}")
    if xtick == 'beat' and beat_resolution is None:
        raise ValueError("`beat_resolution` must be a number when `xtick` "
                         "is 'beat'")
    if ytick not in ('octave', 'pitch', 'off'):
        raise ValueError("`ytick` must be one of {octave', 'pitch', 'off'}")
    if not isinstance(xticklabel, bool):
        raise TypeError("`xticklabel` must be of bool type")
    if yticklabel not in ('auto', 'name', 'number', 'off'):
        raise ValueError("`yticklabel` must be one of {'auto', 'name', "
                         "'number', 'off'}")
    if tick_direction not in ('in', 'out', 'inout'):
        raise ValueError("`tick_direction` must be one of {'in', 'out',"
                         "'inout'}")
    if label not in ('x', 'y', 'both', 'off'):
        raise ValueError("`label` must be one of {'x', 'y', 'both', 'off'}")
    if grid not in ('x', 'y', 'both', 'off'):
        raise ValueError("`grid` must be one of {'x', 'y', 'both', 'off'}")

    # plotting
    if pianoroll.ndim > 2:
        to_plot = pianoroll.transpose(1, 0, 2)
    else:
        to_plot = pianoroll.T
    if (np.issubdtype(pianoroll.dtype, np.bool_)
            or np.issubdtype(pianoroll.dtype, np.floating)):
        ax.imshow(to_plot, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                  origin='lower', interpolation='none')
    elif np.issubdtype(pianoroll.dtype, np.integer):
        ax.imshow(to_plot, cmap=cmap, aspect='auto', vmin=0, vmax=127,
                  origin='lower', interpolation='none')
    else:
        raise TypeError("Unsupported data type for `pianoroll`")

    # tick setting
    if tick_loc is None:
        tick_loc = ('bottom', 'left')
    if xtick == 'auto':
        xtick = 'beat' if beat_resolution is not None else 'step'
    if yticklabel == 'auto':
        yticklabel = 'name' if ytick == 'octave' else 'number'

    if preset == 'plain':
        ax.axis('off')
    elif preset == 'frame':
        ax.tick_params(direction=tick_direction, bottom=False, top=False,
                       left=False, right=False, labelbottom=False,
                       labeltop=False, labelleft=False, labelright=False)
    else:
        ax.tick_params(direction=tick_direction, bottom=('bottom' in tick_loc),
                       top=('top' in tick_loc), left=('left' in tick_loc),
                       right=('right' in tick_loc),
                       labelbottom=(xticklabel != 'off'),
                       labelleft=(yticklabel != 'off'),
                       labeltop=False, labelright=False)

    # x-axis
    if xtick == 'beat' and preset != 'frame':
        num_beat = pianoroll.shape[0]//beat_resolution
        xticks_major = beat_resolution * np.arange(0, num_beat)
        xticks_minor = beat_resolution * (0.5 + np.arange(0, num_beat))
        xtick_labels = np.arange(1, 1 + num_beat)
        ax.set_xticks(xticks_major)
        ax.set_xticklabels('')
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xtick_labels, minor=True)
        ax.tick_params(axis='x', which='minor', width=0)

    # y-axis
    if ytick == 'octave':
        ax.set_yticks(np.arange(0, 128, 12))
        if yticklabel == 'name':
            ax.set_yticklabels(['C{}'.format(i - 2) for i in range(11)])
    elif ytick == 'step':
        ax.set_yticks(np.arange(0, 128))
        if yticklabel == 'name':
            if is_drum:
                ax.set_yticklabels([pretty_midi.note_number_to_drum_name(i)
                                    for i in range(128)])
            else:
                ax.set_yticklabels([pretty_midi.note_number_to_name(i)
                                    for i in range(128)])

    # axis labels
    if label == 'x' or label == 'both':
        if xtick == 'step' or not xticklabel:
            ax.set_xlabel('time (step)')
        else:
            ax.set_xlabel('time (beat)')

    if label == 'y' or label == 'both':
        if is_drum:
            ax.set_ylabel('key name')
        else:
            ax.set_ylabel('pitch')

    # grid
    if grid != 'off':
        ax.grid(axis=grid, color='k', linestyle=grid_linestyle,
                linewidth=grid_linewidth)

    # downbeat boarder
    if downbeats is not None and preset != 'plain':
        for step in downbeats:
            ax.axvline(x=step, color='k', linewidth=1)

def plot_track(track, filepath=None, beat_resolution=None, downbeats=None,
               preset='default', cmap='Blues', xtick='auto', ytick='octave',
               xticklabel=True, yticklabel='auto', tick_loc=None,
               tick_direction='in', label='both', grid='both',
               grid_linestyle=':', grid_linewidth=.5):
    """
    Plot the piano-roll or save a plot of the piano-roll.

    Parameters
    ----------
    filepath :
        The filepath to save the plot. If None, default to save nothing.
    beat_resolution : int
        Resolution of a beat (in time step). Required and only effective
        when `xtick` is 'beat'.
    downbeats : list
        Indices of time steps that contain downbeats., i.e. the first time
        step of a bar.

    preset : {'default', 'plain', 'frame'}
        Preset themes for the plot.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        Colormap to use in :func:`matplotlib.pyplot.imshow`. Default to
        'Blues'. Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        Use beat number or step number as ticks along the x-axis, or
        automatically set to 'beat' when `beat_resolution` is given and set
        to 'step', otherwise. Default to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        Use octave or pitch as ticks along the y-axis. Default to 'octave'.
    xticklabel : bool
        Indicate whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum`
        is True) as tick labels along the y-axis. If 'number', use pitch
        number. If 'auto', set to 'name' when `ytick` is 'octave' and
        'number' when `ytick` is 'pitch'. Default to 'auto'. Only effective
        when `ytick` is not 'off'.
    tick_loc : tuple or list
        List of locations to put ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. If None, default to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        Put ticks inside the axes, outside the axes, or both. Default to
        'in'. Only effective when `xtick` and `ytick` are not both 'off'.
    label : {'x', 'y', 'both', 'off'}
        Add label to the x-axis, y-axis, both or neither. Default to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        Add grid to the x-axis, y-axis, both or neither. Default to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linestyle'
        argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linewidth'
        argument.

    Returns
    -------
    fig : `matplotlib.figure.Figure` object
        A :class:`matplotlib.figure.Figure` object.
    ax : `matplotlib.axes.Axes` object
        A :class:`matplotlib.axes.Axes` object.

    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib package is required for plotting "
                          "supports.")

    fig, ax = plt.subplots()
    plot_pianoroll(ax, track.pianoroll, track.is_drum, beat_resolution,
                   downbeats, preset=preset, cmap=cmap, xtick=xtick,
                   ytick=ytick, xticklabel=xticklabel, yticklabel=yticklabel,
                   tick_loc=tick_loc, tick_direction=tick_direction,
                   label=label, grid=grid, grid_linestyle=grid_linestyle,
                   grid_linewidth=grid_linewidth)

    if filepath is not None:
        plt.savefig(filepath)

    return fig, ax

def plot_multitrack(multitrack, filepath=None, mode='separate',
                    track_label='name', preset='default', cmaps=None,
                    xtick='auto', ytick='octave', xticklabel=True,
                    yticklabel='auto', tick_loc=None, tick_direction='in',
                    label='both', grid='both', grid_linestyle=':',
                    grid_linewidth=.5):
    """
    Plot the piano-rolls or save a plot of them.

    Parameters
    ----------
    filepath : str
        The filepath to save the plot. If None, default to save nothing.
    mode : {'separate', 'stacked', 'hybrid'}
        Plotting modes. Default to 'separate'.

        - In 'separate' mode, all the tracks are plotted separately.
        - In 'stacked' mode, a color is assigned based on `cmaps` to the
            piano-roll of each track and the piano-rolls are stacked and
            plotted as a colored image with RGB channels.
        - In 'hybrid' mode, the drum tracks are merged into a 'Drums' track,
            while the other tracks are merged into an 'Others' track, and the
            two merged tracks are then plotted separately.

    track_label : {'name', 'program', 'family', 'off'}
        Add track name, program name, instrument family name or none as
        labels to the track. When `mode` is 'hybrid', all options other
        than 'off' will label the two track with 'Drums' and 'Others'.
    preset : {'default', 'plain', 'frame'}
        Preset themes for the plot.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmaps :  tuple or list
        List of `matplotlib.colors.Colormap` instances or colormap codes.

        - When `mode` is 'separate', each element will be passed to each
            call of :func:`matplotlib.pyplot.imshow`. Default to ('Blues',
            'Oranges', 'Greens', 'Reds', 'Purples', 'Greys').
        - When `mode` is stacked, a color is assigned based on `cmaps` to
            the piano-roll of each track. Default to ('hsv').
        - When `mode` is 'hybrid', the first (second) element is used in the
            'Drums' ('Others') track. Default to ('Blues', 'Greens').

    xtick : {'auto', 'beat', 'step', 'off'}
        Use beat number or step number as ticks along the x-axis, or
        automatically set to 'beat' when `beat_resolution` is given and set
        to 'step', otherwise. Default to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        Use octave or pitch as ticks along the y-axis. Default to 'octave'.
    xticklabel : bool
        Indicate whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum`
        is True) as tick labels along the y-axis. If 'number', use pitch
        number. If 'auto', set to 'name' when `ytick` is 'octave' and
        'number' when `ytick` is 'pitch'. Default to 'auto'. Only effective
        when `ytick` is not 'off'.
    tick_loc : tuple or list
        List of locations to put ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. If None, default to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        Put ticks inside the axes, outside the axes, or both. Default to
        'in'. Only effective when `xtick` and `ytick` are not both 'off'.
    label : {'x', 'y', 'both', 'off'}
        Add label to the x-axis, y-axis, both or neither. Default to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        Add grid to the x-axis, y-axis, both or neither. Default to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linestyle'
        argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linewidth'
        argument.

    Returns
    -------
    fig : `matplotlib.figure.Figure` object
        A :class:`matplotlib.figure.Figure` object.
    axs : list
        List of :class:`matplotlib.axes.Axes` object.

    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib package is required for plotting "
                          "supports.")

    def get_track_label(track_label, track=None):
        """Convenient function to get track labels"""
        if track_label == 'name':
            return track.name
        elif track_label == 'program':
            return pretty_midi.program_to_instrument_name(track.program)
        elif track_label == 'family':
            return pretty_midi.program_to_instrument_class(track.program)
        elif track is None:
            return track_label

    def add_tracklabel(ax, track_label, track=None):
        """Convenient function for adding track labels"""
        if not ax.get_ylabel():
            return
        ax.set_ylabel(get_track_label(track_label, track) + '\n\n'
                      + ax.get_ylabel())

    multitrack.check_validity()
    if not multitrack.tracks:
        raise ValueError("There is no track to plot")
    if mode not in ('separate', 'stacked', 'hybrid'):
        raise ValueError("`mode` must be one of {'separate', 'stacked', "
                         "'hybrid'}")
    if track_label not in ('name', 'program', 'family', 'off'):
        raise ValueError("`track_label` must be one of {'name', 'program', "
                         "'family'}")

    if cmaps is None:
        if mode == 'separate':
            cmaps = ('Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Greys')
        elif mode == 'stacked':
            cmaps = ('hsv')
        else:
            cmaps = ('Blues', 'Greens')

    num_track = len(multitrack.tracks)
    downbeats = multitrack.get_downbeat_steps()

    if mode == 'separate':
        if num_track > 1:
            fig, axs = plt.subplots(num_track, sharex=True)
        else:
            fig, ax = plt.subplots()
            axs = [ax]

        for idx, track in enumerate(multitrack.tracks):
            now_xticklabel = xticklabel if idx < num_track else False
            plot_pianoroll(axs[idx], track.pianoroll, False,
                           multitrack.beat_resolution, downbeats, preset=preset,
                           cmap=cmaps[idx%len(cmaps)], xtick=xtick, ytick=ytick,
                           xticklabel=now_xticklabel, yticklabel=yticklabel,
                           tick_loc=tick_loc, tick_direction=tick_direction,
                           label=label, grid=grid,
                           grid_linestyle=grid_linestyle,
                           grid_linewidth=grid_linewidth)
            if track_label != 'none':
                add_tracklabel(axs[idx], track_label, track)

        if num_track > 1:
            fig.subplots_adjust(hspace=0)

        if filepath is not None:
            plt.savefig(filepath)

        return (fig, axs)

    elif mode == 'stacked':
        is_all_drum = True
        for track in multitrack.tracks:
            if not track.is_drum:
                is_all_drum = False

        fig, ax = plt.subplots()
        stacked = multitrack.get_stacked_pianorolls()

        colormap = matplotlib.cm.get_cmap(cmaps[0])
        cmatrix = colormap(np.arange(0, 1, 1 / num_track))[:, :3]
        recolored = np.matmul(stacked.reshape(-1, num_track), cmatrix)
        stacked = recolored.reshape(stacked.shape[:2] + (3, ))

        plot_pianoroll(ax, stacked, is_all_drum, multitrack.beat_resolution,
                       downbeats, preset=preset, xtick=xtick, ytick=ytick,
                       xticklabel=xticklabel, yticklabel=yticklabel,
                       tick_loc=tick_loc, tick_direction=tick_direction,
                       label=label, grid=grid, grid_linestyle=grid_linestyle,
                       grid_linewidth=grid_linewidth)

        if track_label != 'none':
            patches = [Patch(color=cmatrix[idx],
                             label=get_track_label(track_label, track))
                       for idx, track in enumerate(multitrack.tracks)]
            plt.legend(handles=patches)

        if filepath is not None:
            plt.savefig(filepath)

        return (fig, [ax])

    elif mode == 'hybrid':
        drums = [i for i, track in enumerate(multitrack.tracks) if track.is_drum]
        others = [i for i in range(len(multitrack.tracks)) if i not in drums]
        merged_drums = multitrack.get_merged_pianoroll(drums)
        merged_others = multitrack.get_merged_pianoroll(others)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        plot_pianoroll(ax1, merged_drums, True, multitrack.beat_resolution,
                       downbeats, preset=preset, cmap=cmaps[0], xtick=xtick,
                       ytick=ytick, xticklabel=xticklabel,
                       yticklabel=yticklabel, tick_loc=tick_loc,
                       tick_direction=tick_direction, label=label, grid=grid,
                       grid_linestyle=grid_linestyle,
                       grid_linewidth=grid_linewidth)
        plot_pianoroll(ax2, merged_others, False, multitrack.beat_resolution,
                       downbeats, preset=preset, cmap=cmaps[1], ytick=ytick,
                       xticklabel=xticklabel, yticklabel=yticklabel,
                       tick_loc=tick_loc, tick_direction=tick_direction,
                       label=label, grid=grid, grid_linestyle=grid_linestyle,
                       grid_linewidth=grid_linewidth)
        fig.subplots_adjust(hspace=0)

        if track_label != 'none':
            add_tracklabel(ax1, 'Drums')
            add_tracklabel(ax2, 'Others')

        if filepath is not None:
            plt.savefig(filepath)

        return (fig, [ax1, ax2])

def save_animation(filepath, pianoroll, window, hop=1, fps=None, is_drum=False,
                   beat_resolution=None, downbeats=None, preset='default',
                   cmap='Blues', xtick='auto', ytick='octave', xticklabel=True,
                   yticklabel='auto', tick_loc=None, tick_direction='in',
                   label='both', grid='both', grid_linestyle=':',
                   grid_linewidth=.5, **kwargs):
    """
    Save a piano-roll to an animation in video or GIF format.

    Parameters
    ----------
    filepath : str
        Path to save the video file.
    pianoroll : np.ndarray
        The piano-roll to be plotted. The values should be in [0, 1] when data
        type is float, and in [0, 127] when data type is integer.

        - For a 2D array, shape=(num_time_step, num_pitch).
        - For a 3D array, shape=(num_time_step, num_pitch, num_channel),
          where channels can be either RGB or RGBA.

    window : int
        Window size to be applied to `pianoroll` for the animation.
    hop : int
        Hop size to be applied to `pianoroll` for the animation.
    fps : int
        Number of frames per second in the resulting video or GIF file.
    is_drum : bool
        Drum indicator. True for drums. False for other instruments. Default
        to False.
    beat_resolution : int
        Resolution of a beat (in time step). Required and only effective
        when `xtick` is 'beat'.
    downbeats : list
        Indices of time steps that contain downbeats., i.e. the first time
        step of a bar.

    preset : {'default', 'plain', 'frame'}
        Preset themes for the plot.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        Colormap to use in :func:`matplotlib.pyplot.imshow`. Default to
        'Blues'. Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        Use beat number or step number as ticks along the x-axis, or
        automatically set to 'beat' when `beat_resolution` is given and set
        to 'step', otherwise. Default to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        Use octave or pitch as ticks along the y-axis. Default to 'octave'.
    xticklabel : bool
        Indicate whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum`
        is True) as tick labels along the y-axis. If 'number', use pitch
        number. If 'auto', set to 'name' when `ytick` is 'octave' and
        'number' when `ytick` is 'pitch'. Default to 'auto'. Only effective
        when `ytick` is not 'off'.
    tick_loc : tuple or list
        List of locations to put ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. If None, default to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        Put ticks inside the axes, outside the axes, or both. Default to
        'in'. Only effective when `xtick` and `ytick` are not both 'off'.
    label : {'x', 'y', 'both', 'off'}
        Add label to the x-axis, y-axis, both or neither. Default to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        Add grid to the x-axis, y-axis, both or neither. Default to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linestyle'
        argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linewidth'
        argument.

    """
    if not HAS_MOVIEPY:
        raise ImportError("moviepy package is required for animation supports.")

    def make_frame(t):
        """Return an image of the frame for time t."""
        fig = plt.gcf()
        ax = plt.gca()
        f_idx = int(t * fps)
        start = hop * f_idx
        end = start + window
        to_plot = transposed[:, start:end]
        extent = (start, end - 1, 0, 127)
        ax.imshow(to_plot, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                  origin='lower', interpolation='none', extent=extent)

        if xtick == 'beat':
            next_major_idx = beat_resolution - start % beat_resolution
            if start % beat_resolution < beat_resolution//2:
                next_minor_idx = beat_resolution//2 - start % beat_resolution
            else:
                next_minor_idx = (beat_resolution//2 - start % beat_resolution
                                  + beat_resolution)
            xticks_major = np.arange(next_major_idx, window, beat_resolution)
            xticks_minor = np.arange(next_minor_idx, window, beat_resolution)
            if end % beat_resolution < beat_resolution//2:
                last_minor_idx = beat_resolution//2 - end % beat_resolution
            else:
                last_minor_idx = (beat_resolution//2 - end % beat_resolution
                                  + beat_resolution)
            xtick_labels = np.arange((start + next_minor_idx)//beat_resolution,
                                     (end + last_minor_idx)//beat_resolution)
            ax.set_xticks(xticks_major)
            ax.set_xticklabels('')
            ax.set_xticks(xticks_minor, minor=True)
            ax.set_xticklabels(xtick_labels, minor=True)
            ax.tick_params(axis='x', which='minor', width=0)

        return mplfig_to_npimage(fig)

    if xtick == 'auto':
        xtick = 'beat' if beat_resolution is not None else 'step'

    fig, ax = plt.subplots()
    plot_pianoroll(ax, pianoroll[:window], is_drum, beat_resolution, downbeats,
                   preset=preset, cmap=cmap, xtick=xtick, ytick=ytick,
                   xticklabel=xticklabel, yticklabel=yticklabel,
                   tick_loc=tick_loc, tick_direction=tick_direction,
                   label=label, grid=grid, grid_linestyle=grid_linestyle,
                   grid_linewidth=grid_linewidth)

    num_frame = int((pianoroll.shape[0] - window) / hop)
    duration = int(num_frame / fps)

    if (np.issubdtype(pianoroll.dtype, np.bool_)
            or np.issubdtype(pianoroll.dtype, np.floating)):
        vmax = 1
    elif np.issubdtype(pianoroll.dtype, np.integer):
        vmax = 127
    else:
        raise TypeError("Unsupported data type for `pianoroll`")
    vmin = 0

    transposed = pianoroll.T
    animation = VideoClip(make_frame, duration=duration)
    if filepath.endswith('.gif'):
        animation.write_gif(filepath, fps, **kwargs)
    else:
        animation.write_videofile(filepath, fps, **kwargs)
    plt.close()

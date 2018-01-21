"""
Module for plotting multi-track and single-track piano-rolls
"""
import numpy as np
import pretty_midi
from matplotlib import pyplot as plt

def plot_pianoroll(ax, pianoroll, lowest=0, is_drum=False, beat_resolution=None,
                   downbeats=None, preset='default', cmap='Blues',
                   xtick='bottom', ytick='left', xticklabel='auto',
                   yticklabel='auto', direction='in', label='both', grid='both',
                   grid_linestyle=':', grid_linewidth=.5):
    """
    Plot a piano-roll given as a numpy array

    Parmeters
    ---------
    ax : matplotlib.axes.Axes object
         The :class:`matplotlib.axes.Axes` object where the piano-roll will
         be plotted on.
    pianoroll : np.ndarray
        The piano-roll to be plotted.
        - For 2D array, shape=(num_time_step, num_pitch).
        - For 3D array, shape=(num_time_step, num_pitch, num_channel), where
          channels can be either RGB or RGBA.
    lowest : int
        Indicate the lowest pitch in the piano-roll. Required when `obj`
        is a numpy array.
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
    xtick : {'bottom', 'top', 'both', 'off'}
        Put ticks along x-axis at top, bottom, both or neither. Default to
        'bottom'.
    ytick : {'left', 'right', 'both', 'off'}
        Put ticks along y-axis at left, right, both or neither. Default to
        'left'.
    xticklabel : {'auto', 'beat', 'step', 'none'}
        Use beat number, step number or neither as tick labels along the
        x-axis, or automatically set to 'beat' when `beat_resolution` is
        given and set to 'step', otherwise. Default to 'auto'. Only
        effective when `xtick` is not 'off'.
    yticklabel : {'auto', 'octave', 'name', 'number', 'none'}
        Use octave name, pitch name or pitch number or none as tick labels
        along the y-axis, or automatically set to 'octave' when `is_drum` is
        False and set to 'name', otherwise. Default to 'auto'. Only
        effective when `ytick` is not 'off'.
    direction : {'in', 'out', 'inout'}
        Put ticks inside the axes, outside the axes, or both. Default to
        'in'. Only effective when `xtick` and `ytick` are not both 'off'.
    label : {'x', 'y', 'both', 'off'}
        Add label to the x-axis, y-axis, both or neither. Default to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        Add grid to the x-axis, y-axis, both or neither. Default to 'both'.
    grid_linestyle : str
        Will be passed to :method:`matplotlib.axes.Axes.grid` as 'linestyle'
        argument.
    grid_linewidth : float
        Will be passed to :method:`matplotlib.axes.Axes.grid` as 'linewidth'
        argument.
    """
    if pianoroll.ndim not in (2, 3):
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")
    if xtick not in ('bottom', 'top', 'both', 'off'):
        raise ValueError("`xtick` must be one of {'bottom', 'top', 'both', "
                         "'off'}")
    if ytick not in ('left', 'right', 'both', 'off'):
        raise ValueError("`ytick` must be one of {'left', 'right', 'both', "
                         "'off'}")
    if xticklabel not in ('auto', 'beat', 'step', 'off'):
        raise ValueError("`xticklabel` must be one of {'auto', 'beat', 'step', "
                         "'none'}")
    if xticklabel is 'beat' and beat_resolution is None:
        raise ValueError("`beat_resolution` must be a number when `xticklabel` "
                         "is 'beat'")
    if yticklabel not in ('auto', 'octave', 'name', 'number', 'off'):
        raise ValueError("`yticklabel` must be one of {'auto', 'octave', "
                         "'name', 'number', 'none'}")
    if direction not in ('in', 'out', 'inout'):
        raise ValueError("`direction` must be one of {'in', 'out', 'inout'}")
    if label not in ('x', 'y', 'both', 'off'):
        raise ValueError("`label` must be one of {'x', 'y', 'both', 'off'}")
    if grid not in ('x', 'y', 'both', 'off'):
        raise ValueError("`grid` must be one of {'x', 'y', 'both', 'off'}")


    # plotting
    if pianoroll.ndim > 2:
        to_plot = pianoroll.transpose(1, 0, 2)
    else:
        to_plot = pianoroll.T
    highest = lowest + pianoroll.shape[1] - 1
    extent = (0, pianoroll.shape[0], lowest, highest)
    ax.imshow(to_plot, cmap=cmap, aspect='auto', vmin=0, vmax=1,
              interpolation='none', extent=extent, origin='lower')

    # tick setting
    if xticklabel == 'auto' and beat_resolution is not None:
        xticklabel = 'beat'
    else:
        xticklabel = 'step'

    if yticklabel == 'auto' and is_drum:
        yticklabel = 'name'
    else:
        yticklabel = 'octave'

    if preset == 'plain':
        ax.axis('off')
    elif preset == 'frame':
        ax.tick_params(direction=direction, bottom='off', top='off', left='off',
                       right='off', labelbottom='off', labeltop='off',
                       labelleft='off', labelright='off')
    else:
        bottom = 'on' if xtick == 'bottom' or xtick == 'both' else 'off'
        top = 'on' if xtick == 'top' or xtick == 'both' else 'off'
        left = 'on' if ytick == 'left' or ytick == 'both' else 'off'
        right = 'on' if ytick == 'right' or ytick == 'both' else 'off'

        labelbottom = 'off' if xticklabel == 'none' else 'on'
        labelleft = 'off' if yticklabel == 'none' else 'on'

        ax.tick_params(direction=direction, bottom=bottom, top=top, left=left,
                       right=right, labelbottom=labelbottom, labeltop='off',
                       labelleft=labelleft, labelright='off')

    # x-axis
    if xticklabel == 'beat':
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
    highest = lowest + pianoroll.shape[1] - 1
    if yticklabel == 'octave':
        ytick_first = lowest + 12 - lowest%12
        ax.set_yticks(np.arange(ytick_first, highest, 12))
        ax.set_yticklabels(['C{}'.format(octave-2)
                            for octave in range(ytick_first//12, highest//12)])
    elif yticklabel == 'name':
        ax.set_yticks(np.arange(lowest, highest))
        if is_drum:
            ax.set_yticklabels([pretty_midi.note_number_to_drum_name(i)
                                for i in range(lowest, highest)])
        else:
            ax.set_yticklabels([pretty_midi.note_number_to_name(i)
                                for i in range(lowest, highest)])

    # axis labels
    if label == 'x' or label == 'both':
        if xticklabel == 'step' or xticklabel == 'off':
            ax.set_xlabel('time (step)')
        else:
            ax.set_xlabel('time (beat)')

    if label == 'y' or label == 'both':
        if (yticklabel == 'name' or is_drum) and yticklabel != 'off':
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

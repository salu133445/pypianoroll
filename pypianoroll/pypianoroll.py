"""
Functions to manipulate multi-track and single-track piano-rolls with
metadata.

Only work for :class:`pianoroll.Multitrack` and :class:`pianoroll.Track`
objects.
"""
from copy import deepcopy
import numpy as np
import pretty_midi
from matplotlib import pyplot as plt
from .track import Track
from .multitrack import Multitrack

def is_pianoroll(arr):
    """
    Return True if the array is a valid piano-roll matrix. Otherwise, return
    False.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("`arr` must be of np.ndarray type")
    if not (np.issubdtype(arr.dtype, np.bool)
            or np.issubdtype(arr.dtype, np.int)
            or np.issubdtype(arr.dtype, np.float)):
        return False
    if arr.ndim != 2:
        return False
    return True

def is_standard_pianoroll(arr):
    """
    Return True if the array is a standard piano-roll matrix that has a
    pitch range under 128. Otherwise, return False.
    """
    if not is_pianoroll(arr):
        return False
    if arr.shape[2] > 128:
        return False
    return True

def binarize(obj, threshold=0):
    """
    Return a copy of the object with binarized piano-roll(s)

    Parameters
    ----------
    threshold : int or float
        Threshold to binarize the piano-roll(s). Default to zero.
    """
    copied = deepcopy(obj)
    copied.binarize(threshold)
    return copied

def clip(obj, lower=0, upper=128):
    """
    Return a copy of the object with piano-roll(s) clipped by a lower bound
    and an upper bound specified by `lower` and `upper`, respectively

    Parameters
    ----------
    lower : int or float
        The lower bound to clip the piano-roll. Default to 0.
    upper : int or float
        The upper bound to clip the piano-roll. Default to 128.
    """
    copied = deepcopy(obj)
    copied.clip(lower, upper)
    return copied

def compress_to_active(obj):
    """
    Return a copy of the object with piano-roll(s) compressed to active
    pitch range(s)
    """
    copied = deepcopy(obj)
    copied.compress_to_active()
    return copied

def copy(obj):
    """Return a copy of the object"""
    copied = deepcopy(obj)
    return copied

def transpose(obj, semitone):
    """
    Return a copy of the object with piano-roll(s) transposed by
    ``semitones`` semitones

    Parameters
    ----------
    semitone : int
        Number of semitones to transpose the piano-roll(s).
    """
    copied = deepcopy(obj)
    copied.lowest_pitch += semitone
    return copied

def trim_trailing_silence(obj):
    """
    Return a copy of the object with trimmed trailing silence of the
    piano-roll(s)
    """
    copied = deepcopy(obj)
    length = copied.get_length()
    copied.pianoroll = copied.pianoroll[:length]
    return copied

def plot(obj, lowest=None, beat_resolution=None, downbeats=None,
         preset='default', xtick='bottom', ytick='left', xticklabel='auto',
         yticklabel='octave', grid='both', grid_linewidth=1.0, label='both',
         **kwargs):
    """
    Plot the piano-roll(s) defined by a numpy array or given in a
    :class:`pypianoroll.Track` or :class:`pypianoroll.Multitrack` object

    Notes
    -----
    If `obj` is a piano-roll, its velocities should be in [0, 1].

    Parmeters
    ---------
    obj : np.ndarray or `pypianoroll.Track` or `pypianoroll.Multitrack`
        The object to be plotted. If a numpy array is provided, shape must
        be (num_time_step, num_pitch).
    lowest : int
        Indicate the lowest pitch in the piano-roll. Required when `obj`
        is a numpy array.
    downbeat : np.ndarray, shape=(num_time_step,), dtype=bool
        Downbeat array that indicates whether the time step contains a
        downbeat, i.e. the first time step of a bar.
    beat_resolution : int
        Resolution of a beat (in time step). Required and only effective
        when `xticklabel` is 'beat'.
    preset : {'default', 'plain', 'frame'}
        Preset themes. In 'default' preset, the ticks and grid are both on.
        In 'frame' preset, the ticks and grid are both off. In 'plain'
        preset, the axes are on. Ignore all other parameters except
        `**kwargs` when using preset other than 'default'.
    xtick : {'bottom', 'top', 'both', 'off'}
        Put ticks along x-axis at top, bottom, both or neither. Default to
        'bottom'.
    ytick : {'left', 'right', 'both', 'off'}
        Put ticks along y-axis at left, right, both or neither. Default to
        'left'.
    xticklabel : {'auto', 'beat', 'step', 'off'}
        Use beat number, step number or neither as tick labels along the
        x-axis, or automatically set to 'beat' when `beat_resolution` is
        given; otherwise, set to 'step'. Default to 'auto'.
    yticklabel : {'octave', 'name', 'number', 'off'}
        Use octave name, pitch name or pitch number or none as tick labels
        along the y-axis. Default to 'octave'.
    grid : {'both', 'x', 'y', 'off'}
        Add grid along the x-axis, y-axis, both or neither. Default to
        'both'.
    label : {'x', 'y', 'both', 'off'}
        True to add labels to the x-axis, y-axis, both or neither. Default
        to 'both'.
    direction : {'in', 'out', 'inout'}
        Put ticks inside the axes, outside the axes, or both. Default to
        'in'.
    **kwargs
        Arbitrary keyword arguments. Will be passed to
        :func:`matplotlib.pyplot.imshow`.
    """

    if isinstance(obj, np.ndarray):
        pianoroll = obj / 127.
        fig, ax = plt.subplots()
        plot_pianoroll(ax, pianoroll, lowest=lowest,
                       beat_resolution=beat_resolution, downbeats=downbeats,
                       preset=preset, xtick=xtick, ytick=ytick,
                       xticklabel=xticklabel, yticklabel=yticklabel,
                       grid=grid, grid_linewidth=grid_linewidth, label=label,
                       **kwargs)
    elif isinstance(obj, Track):
        obj.plot(beat_resolution=beat_resolution, downbeats=downbeats,
                 preset=preset, xtick=xtick, ytick=ytick, xticklabel=xticklabel,
                 yticklabel=yticklabel, grid=grid,
                 grid_linewidth=grid_linewidth, label=label,
                 **kwargs)
    elif isinstance(obj, Multitrack):
        obj.plot(beat_resolution=beat_resolution, downbeats=downbeats,
                 preset=preset, xtick=xtick, ytick=ytick, xticklabel=xticklabel,
                 yticklabel=yticklabel, grid=grid,
                 grid_linewidth=grid_linewidth, label=label,
                 **kwargs)
    else:
        raise TypeError("")

def plot_pianoroll(ax, pianoroll, lowest=0, beat_resolution=None,
                   downbeats=None, preset='default', xtick='bottom',
                   ytick='left', xticklabel='auto', yticklabel='octave',
                   grid='both', grid_linewidth=1.0, label='both', **kwargs):
    """
    Plot a piano-roll given as a numpy array

    Parmeters
    ---------
    ax : matplotlib.axes.Axes object
         The :class:`matplotlib.axes.Axes` object where the piano-roll to be
         plotted on.
    pianoroll : np.ndarray, shape=(num_time_step, num_pitch)
        The piano-roll to be plotted.
    lowest : int
        Indicate the lowest pitch in the piano-roll. Required when `obj`
        is a numpy array.
    beat_resolution : int
        Resolution of a beat (in time step). Required and only effective
        when `xticklabel` is 'beat'.
    downbeat : np.ndarray, shape=(num_time_step,), dtype=bool
        Downbeat array that indicates whether the time step contains a
        downbeat, i.e. the first time step of a bar.
    preset : {'default', 'plain', 'frame'}
        Preset themes. In 'default' preset, the ticks and grid are both on.
        In 'frame' preset, the ticks and grid are both off. In 'plain'
        preset, the axes are on. Ignore all other parameters except
        `**kwargs` when using preset other than 'default'.
    xtick : {'bottom', 'top', 'both', 'off'}
        Put ticks along x-axis at top, bottom, both or neither. Default to
        'bottom'.
    ytick : {'left', 'right', 'both', 'off'}
        Put ticks along y-axis at left, right, both or neither. Default to
        'left'.
    xticklabel : {'auto', 'beat', 'step', 'off'}
        Use beat number, step number or neither as tick labels along the
        x-axis, or automatically set to 'beat' when `beat_resolution` is
        given; otherwise, set to 'step'. Default to 'auto'.
    yticklabel : {'octave', 'name', 'number', 'off'}
        Use octave name, pitch name or pitch number or none as tick labels
        along the y-axis. Default to 'octave'.
    grid : {'both', 'x', 'y', 'off'}
        Add grid along the x-axis, y-axis, both or neither. Default to
        'both'.
    label : {'x', 'y', 'both', 'off'}
        True to add labels to the x-axis, y-axis, both or neither. Default
        to 'both'.
    direction : {'in', 'out', 'inout'}
        Put ticks inside the axes, outside the axes, or both. Default to
        'in'.
    **kwargs
        Arbitrary keyword arguments. Will be passed to
        :func:`matplotlib.pyplot.imshow`.

    See Also
    --------
    :func:`plot_track`, :func:`plot_multitrack`
    """
    if xticklabel is 'beat' and beat_resolution is None:
        raise ValueError("`beat_resolution` must be a number when `xticklabel` "
                         "is 'beat'")

    if 'cmap' not in kwargs:
        if preset == 'plain':
            kwargs['cmap'] = 'binary'
        else:
            kwargs['cmap'] = 'Blues'
    if 'extent' not in kwargs:
        highest = lowest + pianoroll.shape[1] - 1
        kwargs['extent'] = (0, pianoroll.shape[0], lowest, highest)
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'
    if 'vmin' not in kwargs:
        kwargs['vmin'] = 0
    if 'vmax' not in kwargs:
        kwargs['vmax'] = 1

    normalized = pianoroll.T
    ax.imshow(normalized, **kwargs)

    if preset == 'plain':
        ax.axis('off')

    if preset == 'frame':
        xtick = 'off'
        ytick = 'off'
        xticklabel = 'off'
        yticklabel = 'off'
        grid = 'off'
        label = 'off'

    bottom = 'on' if xtick == 'bottom' or xtick == 'both' else 'off'
    top = 'on' if xtick == 'top' or xtick == 'both' else 'off'
    left = 'on' if ytick == 'left' or ytick == 'both' else 'off'
    right = 'on' if ytick == 'right' or ytick == 'both' else 'off'

    labelbottom = 'off' if xticklabel == 'off' else 'on'
    labelleft = 'off' if yticklabel == 'off' else 'on'

    ax.tick_params(direction='in', bottom=bottom, top=top, left=left,
                    right=right, labelbottom=labelbottom, labeltop='off',
                    labelleft=labelleft, labelright='off')

    auto_beat = (xticklabel == 'auto' and beat_resolution is not None)
    if xticklabel == 'beat' or auto_beat:
        num_beat = pianoroll.shape[1]//beat_resolution - 1
        xticks_major = beat_resolution * np.arange(0, num_beat)
        xticks_minor = beat_resolution * (0.5 + np.arange(0, num_beat))
        xtick_labels = np.arange(1, 1 + num_beat)
        ax.set_xticks(xticks_major)
        ax.set_xticklabels('')
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xtick_labels, minor=True)
        ax.tick_params(axis='x', which='minor', width=0)

    if label == 'x' or label == 'both':
        if xticklabel == 'beat' or auto_beat:
            ax.set_xlabel('time (beat)')
        else:
            ax.set_xlabel('time (step)')

    ytick_first = lowest + 12 - lowest%12
    highest = lowest + pianoroll.shape[1] - 1
    ytick_last = highest - lowest%12

    if yticklabel == 'octave':
        yticks = np.arange(ytick_first, ytick_last, 12)
        ytick_labels = ['C{}'.format(octave-2)
                        for octave in range(ytick_first//12, ytick_last//12+1)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
    else:
        yticks = np.arange(ytick_first, ytick_last)
        ax.set_yticks(yticks)
        if yticklabel == 'note':
            ytick_labels = [pretty_midi.note_number_to_name(i) for i in yticks]
            ax.set_yticklabels(ytick_labels)

    if label == 'y' or label == 'both':
        if yticklabel == 'octave' or 'note':
            ax.set_ylabel('pitch')
        else:
            ax.set_ylabel('pitch number')

    if grid == 'x' or grid == 'both':
        ax.grid(axis='x', linestyle=':', color='k', linewidth=grid_linewidth)
    if grid == 'y' or grid == 'both':
        ax.grid(axis='y', linestyle=':', color='k', linewidth=grid_linewidth)

    if downbeats is not None and preset != 'plain':
        for step in downbeats:
            ax.axvline(x=step, color='k', linewidth=1)

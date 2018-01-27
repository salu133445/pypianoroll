"""Module for plotting multi-track and single-track piano-rolls

"""
import numpy as np
import pretty_midi
from matplotlib import pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def plot_pianoroll(ax, pianoroll, is_drum=False, beat_resolution=None,
                   downbeats=None, normalization='standard', preset='default',
                   cmap='Blues', tick_loc=None, xtick='auto', ytick='octave',
                   xticklabel='on', yticklabel='auto', direction='in',
                   label='both', grid='both', grid_linestyle=':',
                   grid_linewidth=.5):
    """
    Plot a piano-roll given as a numpy array

    Parameters
    ----------
    ax : matplotlib.axes.Axes object
         The :class:`matplotlib.axes.Axes` object where the piano-roll will
         be plotted on.
    pianoroll : np.ndarray
        The piano-roll to be plotted. The values should be in [0, 1] when
        `normalized` is False.

        - For 2D array, shape=(num_time_step, num_pitch).
        - For 3D array, shape=(num_time_step, num_pitch, num_channel), where
          channels can be either RGB or RGBA.

    is_drum : bool
        Drum indicator. True for drums. False for other instruments. Default
        to False.
    beat_resolution : int
        Resolution of a beat (in time step). Required and only effective
        when `xticklabel` is 'beat'.
    downbeats : list
        Indices of time steps that contain downbeats., i.e. the first time
        step of a bar.
    normalization : {'standard', 'auto', 'none'}
        The normalization method to apply to the piano-roll. Default to
        'standard'. If `pianoroll` is binarized, use 'none' anyway.

        - For 'standard' normalization, the normalized values are given by
          N = P / 128, where P, N is the original and normalized piano-roll,
          respectively
        - For 'auto' normalization, the normalized values are given by
          N = (P - m) / (M - m), where P, N is the original and normalized
          piano-roll, respectively, and M, m is the maximum and minimum of
          the original piano-roll, respectively.
        - If 'none', no normalization will be applied to the piano-roll. In
          this case, the values of `pianoroll` should be in [0, 1] in order
          to plot it correctly.

    preset : {'default', 'plain', 'frame'}
        Preset themes for the plot.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        Colormap to use in :func:`matplotlib.pyplot.imshow`. Default to
        'Blues'. Only effective when `pianoroll` is 2D.
    tick_loc : tuple or list
        List of locations to put ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. If None, default to ('bottom', 'left').
    xtick : {'auto', 'beat', 'step', 'off'}
        Use beat number or step number as ticks along the x-axis, or
        automatically set to 'beat' when `beat_resolution` is given and set
        to 'step', otherwise. Default to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        Use octave or pitch as ticks along the y-axis. Default to 'octave'.
    xticklabel : {'on', 'off'}
        Indicate whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum`
        is True) as tick labels along the y-axis. If 'number', use pitch
        number. If 'auto', set to 'name' when `ytick` is 'octave' and
        'number' when `ytick` is 'pitch'. Default to 'auto'. Only effective
        when `ytick` is not 'off'.
    direction : {'in', 'out', 'inout'}
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
    if pianoroll.ndim not in (2, 3):
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")
    if pianoroll.shape[1] != 128:
        raise ValueError("The shape of `pianoroll` must be (num_time_step, "
                         "128)")
    if xtick not in ('auto', 'beat', 'step', 'off'):
        raise ValueError("`xtick` must be one of {'auto', 'beat', 'step', "
                         "'none'}")
    if xtick is 'beat' and beat_resolution is None:
        raise ValueError("`beat_resolution` must be a number when `xtick` "
                         "is 'beat'")
    if ytick not in ('octave', 'pitch', 'off'):
        raise ValueError("`ytick` must be one of {octave', 'pitch', 'off'}")
    if xticklabel not in ('on', 'off'):
        raise ValueError("`xticklabel` must be 'on' or 'off'")
    if yticklabel not in ('auto', 'name', 'number', 'off'):
        raise ValueError("`yticklabel` must be one of {'auto', 'name', "
                         "'number', 'off'}")
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
    if normalization == 'standard':
        to_plot = to_plot / 128.
    elif normalization == 'auto':
        max_value = np.max(to_plot)
        min_value = np.min(to_plot)
        to_plot = to_plot - min_value / (max_value - min_value)
    ax.imshow(to_plot, cmap=cmap, aspect='auto', vmin=0, vmax=1, origin='lower',
              interpolation='none')

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
        ax.tick_params(direction=direction, bottom='off', top='off', left='off',
                       right='off', labelbottom='off', labeltop='off',
                       labelleft='off', labelright='off')
    else:
        labelbottom = 'on' if xticklabel != 'off' else 'off'
        labelleft = 'on' if yticklabel != 'off' else 'off'

        ax.tick_params(direction=direction, bottom=('bottom' in tick_loc),
                       top=('top' in tick_loc), left=('left' in tick_loc),
                       right=('right' in tick_loc), labelbottom=labelbottom,
                       labeltop='off', labelleft=labelleft, labelright='off')

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
        if xtick == 'step' or xticklabel == 'off':
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

def save_animation(filepath, pianoroll, window, hop=1, fps=None, is_drum=False,
                   beat_resolution=None, downbeats=None,
                   normalization='standard', preset='default', cmap='Blues',
                   tick_loc=None, xtick='auto', ytick='octave', xticklabel='on',
                   yticklabel='auto', direction='in', label='both', grid='both',
                   grid_linestyle=':', grid_linewidth=.5, **kwargs):
    """
    Save a piano-roll to an animation in video or GIF format.

    Parameters
    ----------
    filepath : str
        Path to save the video file.
    pianoroll : np.ndarray
        The piano-roll to be plotted. The values should be in [0, 1] when
        `normalized` is False.

        - For 2D array, shape=(num_time_step, num_pitch).
        - For 3D array, shape=(num_time_step, num_pitch, num_channel), where
          channels can be either RGB or RGBA.

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
        when `xticklabel` is 'beat'.
    downbeats : list
        Indices of time steps that contain downbeats., i.e. the first time
        step of a bar.
    normalization : {'standard', 'auto', 'none'}
        The normalization method to apply to the piano-roll. Default to
        'standard'. If `pianoroll` is binarized, use 'none' anyway.

        - For 'standard' normalization, the normalized values are given by
          N = P / 128, where P, N is the original and normalized piano-roll,
          respectively
        - For 'auto' normalization, the normalized values are given by
          N = (P - m) / (M - m), where P, N is the original and normalized
          piano-roll, respectively, and M, m is the maximum and minimum of the
          original piano-roll, respectively.
        - If 'none', no normalization will be applied to the piano-roll. In
          this case, the values of `pianoroll` should be in [0, 1] in order to
          plot it correctly.

    preset : {'default', 'plain', 'frame'}
        Preset themes for the plot.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        Colormap to use in :func:`matplotlib.pyplot.imshow`. Default to
        'Blues'. Only effective when `pianoroll` is 2D.
    tick_loc : tuple or list
        List of locations to put ticks. Availables elements are 'bottom',
        'top', 'left' and 'right'. If None, default to ('bottom', 'left').
    xtick : {'auto', 'beat', 'step', 'off'}
        Use beat number or step number as ticks along the x-axis, or
        automatically set to 'beat' when `beat_resolution` is given and set
        to 'step', otherwise. Default to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        Use octave or pitch as ticks along the y-axis. Default to 'octave'.
    xticklabel : {'on', 'off'}
        Indicate whether to add tick labels along the x-axis. Only effective
        when `xtick` is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum`
        is True) as tick labels along the y-axis. If 'number', use pitch
        number. If 'auto', set to 'name' when `ytick` is 'octave' and
        'number' when `ytick` is 'pitch'. Default to 'auto'. Only effective
        when `ytick` is not 'off'.
    direction : {'in', 'out', 'inout'}
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
    def make_frame(t):
        """Return an image of the frame for time t"""
        fig = plt.gcf()
        ax = plt.gca()
        f_idx = int(t * fps)
        start = hop * f_idx
        end = start + window
        to_plot = pianoroll[start:end].T / 128.
        extent = (start, end - 1, 0, 127)
        ax.imshow(to_plot, cmap=cmap, aspect='auto', vmin=0, vmax=1,
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
    plot_pianoroll(ax, pianoroll[:window], is_drum=is_drum,
                   beat_resolution=beat_resolution, downbeats=downbeats,
                   normalization=normalization, preset=preset, cmap=cmap,
                   tick_loc=tick_loc, xtick=xtick, ytick=ytick,
                   xticklabel=xticklabel, yticklabel=yticklabel,
                   direction=direction, label='both', grid='both',
                   grid_linestyle=':', grid_linewidth=.5)

    num_frame = int((pianoroll.shape[0] - window) / hop)
    duration = int(num_frame / fps)
    animation = VideoClip(make_frame, duration=duration)
    if filepath.endswith('.gif'):
        animation.write_gif(filepath, fps, kwargs)
    else:
        animation.write_videofile(filepath, fps, kwargs)
    plt.close()

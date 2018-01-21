"""
Class for single-track piano-rolls with metadata.
"""
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from .plot import plot_pianoroll

class Track(object):
    """
    A single-track piano-roll container

    Attributes
    ----------
    pianoroll : np.ndarray, shape=(num_time_step, num_pitch)
        Piano-roll matrix. First dimension represents time. Second dimension
        represents pitch. The lowest pitch is given by `lowest`.
    program: int
        Program number according to General MIDI specification. Available
        values are 0 to 127. Default to 0 (Acoustic Grand Piano).
    is_drum : bool
        Drum indicator. True for drums. False for other instruments.
    name : str
        Name of the track.
    lowest : int
        Indicate the lowest pitch in the piano-roll.
    """

    def __init__(self, pianoroll=None, program=0, is_drum=False, name='unknown',
                 lowest=0):
        """
        Initialize by assigning attributes

        Parameters
        ----------
        pianoroll : np.ndarray, shape=(num_time_step, num_pitch)
            Piano-roll matrix. First dimension represents time. Second dimension
            represents pitch. The lowest pitch is given by `lowest`. Available
            datatypes are bool, int, float.
        program: int
            Program number according to General MIDI specification [1].
            Available values are 0 to 127. Default to 0 (Acoustic Grand Piano).
        is_drum : bool
            Drum indicator. True for drums. False for other instruments. Default
            to False.
        name : str
            Name of the track. Default to 'unknown'.
        lowest : int
            Indicate the lowest pitch of the piano-roll. Default to zero.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set
        """
        if pianoroll is None:
            self.pianoroll = np.zeros((0, 0), bool)
        else:
            self.pianoroll = pianoroll
        self.program = program
        self.is_drum = is_drum
        self.name = name
        self.lowest = lowest

        self.check_validity()

    def __repr__(self):
        return ("Track(pianoroll={}, program={}, is_drum={}, name={}, "
                "lowest={}".format(self.program, self.is_drum, self.name,
                                   self.lowest, self.pianoroll.__str__))

    def __str__(self):
        return ("program : {},\nis_drum : {},\nname : {},\nlowest : {},\n"
                "pianoroll :\n{}".format(self.program, self.is_drum, self.name,
                                         self.lowest, self.pianoroll.__str__))

    def binarize(self, threshold=0):
        """
        Binarize the piano-roll. Do nothing if the piano-roll is already
        binarized

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano-rolls. Default to zero.

        Examples
        --------
        >>> pianoroll = np.arange(24).reshape((4,6))
        >>> pianoroll
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17],
               [18, 19, 20, 21, 22, 23]])
        >>> track = pypianoroll.Track(pianoroll)
        >>> track.binarize(10)
        >>> track.pianoroll
        array([[False, False, False, False, False, False],
               [False, False, False, False, False,  True],
               [ True,  True,  True,  True,  True,  True],
               [ True,  True,  True,  True,  True,  True]], dtype=bool)
        """
        if not self.is_binarized():
            self.pianoroll = (self.pianoroll > threshold)

    def check_validity(self):
        """"Raise error if any invalid attribute found"""
        # pianoroll
        if not isinstance(self.pianoroll, np.ndarray):
            raise TypeError("`pianoroll` must be of np.ndarray type")
        if not (np.issubdtype(self.pianoroll.dtype, np.bool)
                or np.issubdtype(self.pianoroll.dtype, np.int)
                or np.issubdtype(self.pianoroll.dtype, np.float)):
            raise TypeError("Data type of `pianoroll` must be one of bool, int "
                            "and float.")
        if self.pianoroll.ndim != 2:
            raise ValueError("`pianoroll` must be a 2D numpy array")
        # program
        if not isinstance(self.program, int):
            raise TypeError("`program` must be of int type")
        if self.program < 0 or self.program > 127:
            raise ValueError("`program` must be in 0 to 127")
        # is_drum
        if not isinstance(self.is_drum, bool):
            raise TypeError("`is_drum` must be of boolean type")
        # lowest
        if not isinstance(self.lowest, int):
            raise TypeError("`lowest` must be of int type")
        if self.lowest < 0:
            raise ValueError("`lowest` must be a non-negative integer")
        # name
        if not isinstance(self.name, str):
            raise TypeError("`name` must be of str type")

    def clip(self, lower=0, upper=128):
        """
        Clip the piano-roll by an lower bound and an upper bound specified by
        `lower` and `upper`, respectively

        Parameters
        ----------
        lower : int or float
            The lower bound to clip the piano-roll. Default to 0.
        upper : int or float
            The upper bound to clip the piano-roll. Default to 128.

        Examples
        --------
        >>> pianoroll = np.arange(24).reshape((4,6))
        >>> pianoroll
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17],
               [18, 19, 20, 21, 22, 23]])
        >>> track = pypianoroll.Track(pianoroll)
        >>> track.clip(5, 15)
        >>> track.pianoroll
        array([[ 5,  5,  5,  5,  5,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 15, 15],
               [15, 15, 15, 15, 15, 15]])
        """
        self.pianoroll = self.pianoroll.clip(lower, upper)

    def compress_to_active(self):
        """Compress the piano-roll to active pitch range"""
        lowest, highest = self.get_pitch_range(True)
        self.pianoroll = self.pianoroll[:, lowest:highest + 1]
        self.lowest += lowest

    def copy(self):
        """
        Return a copy of the object

        Returns
        -------
        copied : `pypianoroll.Track` object
            A copy of the object.
        """
        copied = deepcopy(self)
        return copied

    def expand(self, lowest=0, highest=127):
        """
        Expand or compress the piano-roll to a pitch range specified by
        `lowest` and `highest`

        Parameters
        ----------
        lowest : int or float
            The lowest pitch of the expanded piano-roll.
        highest : int or float
            The highest pitch of the expanded piano-roll.

        Examples
        --------
        >>> pianoroll = np.ones((96, 40))
        >>> track = pypianoroll.Track(pianoroll, lowest=10)
        >>> track.expand(0, 127)
        >>> track.lowest
        0
        >>> track.pianoroll.shape
        (96, 128)
        >>> track.expand(31, 60)
        >>> track.lowest
        31
        >>> track.pianoroll.shape
        (96, 30)
        """
        if self.lowest > lowest:
            to_pad = self.lowest - lowest
            self.pianoroll = np.pad(self.pianoroll, ((0, 0), (to_pad, 0)),
                                    'constant')
        elif self.lowest < lowest:
            self.pianoroll = self.pianoroll[:, (lowest - self.lowest):]

        self.lowest = lowest

        pitch_axis_length = highest - lowest + 1
        if self.pianoroll.shape[1] < pitch_axis_length:
            to_pad = pitch_axis_length - self.pianoroll.shape[1]
            self.pianoroll = np.pad(self.pianoroll, ((0, 0), (0, to_pad)),
                                    'constant')
        elif self.pianoroll.shape[1] > pitch_axis_length:
            self.pianoroll = self.pianoroll[:, :pitch_axis_length]

    def get_pianoroll(self):
        """
        Return a copy of the piano-roll

        Returns
        -------
        copied :
            A copy of the piano-roll.
        lowest : int
            Indicate the lowest pitch in the piano-roll.
        """
        copied = np.copy(self.pianoroll)
        return copied, self.lowest

    def get_length(self):
        """
        Return the length of the piano-roll without trailing silence (in time
        step)

        Returns
        -------
        length : int
            Length of the piano-roll without trailing silence (in time step).
        """
        non_zero_steps = np.any((self.pianoroll > 0), axis=1)
        inv_last_non_zero_step = np.argmax(np.flip(non_zero_steps, axis=0))
        length = self.pianoroll.shape[0] - inv_last_non_zero_step
        return length

    def get_pitch_range(self, relative=False):
        """
        Return the pitch range in tuple (lowest, highest)

        Parameters
        ----------
        relative : bool
            True to return the relative pitch range , i.e. the corresponding
            indices in the pitch axis of the piano-roll. False to return the
            absolute pitch range. Default to False.

        Returns
        -------
        lowest : int
            Indicate the lowest pitch in the piano-roll.
        highest : int
            Indicate the highest pitch in the piano-roll.
        """
        lowest = 0
        while not np.any(self.pianoroll[:, lowest]):
            lowest += 1

        highest = self.pianoroll.shape[1] - 1
        while not np.any(self.pianoroll[:, highest]):
            highest -= 1

        if not relative:
            lowest = self.lowest + lowest
            highest = self.lowest + highest

        return lowest, highest

    def is_binarized(self):
        """
        Return True if the piano-roll is already binarized. Otherwise, return
        False

        Returns
        -------
        is_binarized : bool
            True if the piano-roll is already binarized; otherwise, False.
        """
        is_binarized = (self.pianoroll.dtype == bool)
        return is_binarized

    def plot(self, filepath=None, beat_resolution=None, downbeats=None,
             preset='default', cmap='Blues', xtick='bottom', ytick='left',
             xticklabel='auto', yticklabel='auto', direction='in', label='both',
             grid='both', grid_linestyle=':', grid_linewidth=.5):
        """
        Plot the piano-roll or save a plot of the piano-roll.

        Parameters
        ----------
        filepath :
            The filepath to save the plot. If None, default to save nothing.
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

        Returns
        -------
        fig : `matplotlib.figure.Figure` object
            A :class:`matplotlib.figure.Figure` object.
        ax : `matplotlib.axes.Axes` object
            A :class:`matplotlib.axes.Axes` object.
        """
        _, ax = plt.subplots()
        plot_pianoroll(ax, self.pianoroll, self.lowest, self.is_drum,
                       beat_resolution=beat_resolution, downbeats=downbeats,
                       preset=preset, cmap=cmap, xtick=xtick, ytick=ytick,
                       xticklabel=xticklabel, yticklabel=yticklabel,
                       direction=direction, label=label, grid=grid,
                       grid_linestyle=grid_linestyle,
                       grid_linewidth=grid_linewidth)

        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath)

    def transpose(self, semitone):
        """
        Transpose the piano-roll by `semitones` semitones

        Parameters
        ----------
        semitone : int
            Number of semitones transpose the piano-roll.

        Examples
        --------
        >>> pianoroll = np.random.randint(0, 127, (96, 128))
        >>> track = pypianoroll.Track(pianoroll)
        >>> track.lowest
        0
        >>> track.transpose(10)
        >>> track.lowest
        10
        """
        self.lowest += semitone

    def trim_trailing_silence(self):
        """Trim the trailing silence of the piano-roll"""
        length = self.get_length()
        self.pianoroll = self.pianoroll[:length]

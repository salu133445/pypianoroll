"""Class for single-track piano-rolls with metadata.

"""
from copy import deepcopy
from six import string_types
import numpy as np
from matplotlib import pyplot as plt
from .plot import plot_pianoroll

class Track(object):
    """
    A single-track piano-roll container.

    Attributes
    ----------
    pianoroll : np.ndarray, shape=(num_time_step, 128)
        Piano-roll matrix. First dimension represents time. Second dimension
        represents pitch.
    program: int
        Program number according to General MIDI specification. Available
        values are 0 to 127. Default to 0 (Acoustic Grand Piano).
    is_drum : bool
        Drum indicator. True for drums. False for other instruments.
    name : str
        Name of the track.

    """

    def __init__(self, pianoroll=None, program=0, is_drum=False,
                 name='unknown'):
        """
        Initialize the object by assigning attributes.

        Parameters
        ----------
        pianoroll : np.ndarray, shape=(num_time_step, 128)
            Piano-roll matrix. First dimension represents time. Second dimension
            represents pitch. Available data types are bool, int, float.
        program: int
            Program number according to General MIDI specification [1].
            Available values are 0 to 127. Default to 0 (Acoustic Grand Piano).
        is_drum : bool
            Drum indicator. True for drums. False for other instruments. Default
            to False.
        name : str
            Name of the track. Default to 'unknown'.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set

        """
        if pianoroll is None:
            self.pianoroll = np.zeros((0, 128), bool)
        else:
            self.pianoroll = pianoroll
        self.program = program
        self.is_drum = is_drum
        self.name = name

        self.check_validity()

    def __getitem__(self, val):
        return Track(self.pianoroll[val], self.program, self.is_drum, self.name)

    def __repr__(self):
        return ("Track(pianoroll={}, program={}, is_drum={}, name={})"
                .format(repr(self.pianoroll), self.program, self.is_drum,
                        self.name))

    def __str__(self):
        return ("pianoroll :\n{},\nprogram : {},\nis_drum : {},\nname : {}"
                .format(str(self.pianoroll), self.program, self.is_drum,
                        self.name))

    def assign_constant(self, value):
        """
        Assign a constant value to all nonzeros in the piano-roll. If the
        piano-roll is not binarized, its data type will be preserved. If the
        piano-roll is binarized, it will be casted to the type of `value`.

        Arguments
        ---------
        value : int or float
            The constant value to be assigned to the nonzeros of the piano-roll.

        """
        if self.is_binarized():
            self.pianoroll = self.pianoroll * value
            return
        self.pianoroll[self.pianoroll.nonzero()] = value

    def binarize(self, threshold=0):
        """
        Binarize the piano-roll. Do nothing if the piano-roll is already
        binarized.

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano-rolls. Default to zero.

        """
        if not self.is_binarized():
            self.pianoroll = (self.pianoroll > threshold)

    def check_validity(self):
        """"Raise error if any invalid attribute found."""
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
        if self.pianoroll.shape[1] != 128:
            raise ValueError("The shape of `pianoroll` must be (num_time_step, "
                             "128)")
        # program
        if not isinstance(self.program, int):
            raise TypeError("`program` must be of int type")
        if self.program < 0 or self.program > 127:
            raise ValueError("`program` must be in 0 to 127")
        # is_drum
        if not isinstance(self.is_drum, bool):
            raise TypeError("`is_drum` must be of boolean type")
        # name
        if not isinstance(self.name, string_types):
            raise TypeError("`name` must be of string type")

    def clip(self, lower=0, upper=127):
        """
        Clip the piano-roll by an lower bound and an upper bound specified by
        `lower` and `upper`, respectively.

        Parameters
        ----------
        lower : int or float
            The lower bound to clip the piano-roll. Default to 0.
        upper : int or float
            The upper bound to clip the piano-roll. Default to 127.

        """
        self.pianoroll = self.pianoroll.clip(lower, upper)

    def copy(self):
        """
        Return a copy of the object.

        Returns
        -------
        copied : `pypianoroll.Track` object
            A copy of the object.

        """
        copied = deepcopy(self)
        return copied

    def pad(self, pad_length):
        """
        Pad the piano-roll with zeros at the end along the time axis.

        Parameters
        ----------
        pad_length : int
            The length to pad along the time axis with zeros.

        """
        self.pianoroll = np.pad(self.pianoroll, ((0, pad_length), (0, 0)),
                                'constant')

    def pad_to_multiple(self, factor):
        """
        Pad the piano-roll with zeros at the end along the time axis with the
        minimal length that make the length of the resulting piano-roll a
        multiple of `factor`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting piano-roll will be
            a multiple of.

        """
        to_pad = factor - self.pianoroll.shape[0]%factor
        self.pianoroll = np.pad(self.pianoroll, ((0, to_pad), (0, 0)),
                                'constant')

    def get_pianoroll_copy(self):
        """
        Return a copy of the piano-roll.

        Returns
        -------
        copied :
            A copy of the piano-roll.

        """
        copied = np.copy(self.pianoroll)
        return copied

    def get_active_length(self):
        """
        Return the active length (i.e. without trailing silence) of the
        piano-roll (in time step).

        Returns
        -------
        active_length : int
            Length of the piano-roll without trailing silence (in time step).

        """
        non_zero_steps = np.any((self.pianoroll > 0), axis=1)
        inv_last_non_zero_step = np.argmax(np.flip(non_zero_steps, axis=0))
        active_length = self.pianoroll.shape[0] - inv_last_non_zero_step
        return active_length

    def get_active_pitch_range(self):
        """
        Return the active pitch range in tuple (lowest, highest).

        Parameters
        ----------
        relative : bool
            True to return the relative pitch range , i.e. the corresponding
            indices in the pitch axis of the piano-roll. False to return the
            absolute pitch range. Default to False.

        Returns
        -------
        lowest : int
            The lowest active pitch in the piano-roll.
        highest : int
            The highest active pitch in the piano-roll.

        """
        if self.pianoroll.shape[1] < 1:
            raise ValueError("Cannot compute the active pitch range for an "
                             "empty piano-roll")
        lowest = 0
        highest = 127
        while lowest < highest:
            if np.any(self.pianoroll[:, lowest]):
                break
            lowest += 1
        if lowest == highest:
            raise ValueError("Cannot compute the active pitch range for an "
                             "empty piano-roll")
        while not np.any(self.pianoroll[:, highest]):
            highest -= 1

        return lowest, highest

    def is_binarized(self):
        """
        Return True if the piano-roll is already binarized. Otherwise, return
        False.

        Returns
        -------
        is_binarized : bool
            True if the piano-roll is already binarized; otherwise, False.

        """
        is_binarized = np.issubdtype(self.pianoroll.dtype, np.bool)
        return is_binarized

    def plot(self, filepath=None, beat_resolution=None, downbeats=None,
             normalization='standard', preset='default', cmap='Blues',
             tick_loc=None, xtick='auto', ytick='octave', xticklabel='on',
             yticklabel='auto', direction='in', label='both', grid='both',
             grid_linestyle=':', grid_linewidth=.5):
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

        Returns
        -------
        fig : `matplotlib.figure.Figure` object
            A :class:`matplotlib.figure.Figure` object.
        ax : `matplotlib.axes.Axes` object
            A :class:`matplotlib.axes.Axes` object.

        """
        fig, ax = plt.subplots()
        if self.is_binarized():
            normalization = 'none'
        plot_pianoroll(ax, self.pianoroll, self.is_drum,
                       beat_resolution=beat_resolution, downbeats=downbeats,
                       normalization=normalization, preset=preset, cmap=cmap,
                       tick_loc=tick_loc, xtick=xtick, ytick=ytick,
                       xticklabel=xticklabel, yticklabel=yticklabel,
                       direction=direction, label=label, grid=grid,
                       grid_linestyle=grid_linestyle,
                       grid_linewidth=grid_linewidth)

        if filepath is not None:
            plt.savefig(filepath)

        return fig, ax

    def transpose(self, semitone):
        """
        Transpose the piano-roll by a certain semitones, where positive
        values are for higher key, while negative values are for lower key.

        Parameters
        ----------
        semitone : int
            Number of semitones to transpose the piano-roll.

        """
        if semitone > 0 and semitone < 128:
            self.pianoroll[:, semitone:] = self.pianoroll[:, :(128 - semitone)]
            self.pianoroll[:, :semitone] = 0
        elif semitone < 0 and semitone > -128:
            self.pianoroll[:, :(128 + semitone)] = self.pianoroll[:, -semitone:]
            self.pianoroll[:, (128 + semitone):] = 0

    def trim_trailing_silence(self):
        """Trim the trailing silence of the piano-roll."""
        length = self.get_active_length()
        self.pianoroll = self.pianoroll[:length]

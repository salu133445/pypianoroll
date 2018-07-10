"""Class for single-track piano-rolls with metadata.

"""
from __future__ import absolute_import, division, print_function
from copy import deepcopy
from six import string_types
import numpy as np
from pypianoroll.plot import plot_track

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

    def assign_constant(self, value, dtype=None):
        """
        Assign a constant value to all nonzeros in the piano-roll. If the
        piano-roll is not binarized, its data type will be preserved. If the
        piano-roll is binarized, it will be casted to the type of `value`.

        Arguments
        ---------
        value : int or float
            The constant value to be assigned to the nonzeros of the piano-roll.

        """
        if not self.is_binarized():
            self.pianoroll[self.pianoroll.nonzero()] = value
            return
        if dtype is None:
            if isinstance(value, int):
                dtype = int
            elif isinstance(value, float):
                dtype = float
        nonzero = self.pianoroll.nonzero()
        self.pianoroll = np.zeros(self.pianoroll.shape, dtype)
        self.pianoroll[nonzero] = value

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
        if not (np.issubdtype(self.pianoroll.dtype, np.bool_)
                or np.issubdtype(self.pianoroll.dtype, np.number)):
            raise TypeError("Data type of `pianoroll` must be np.bool_ or a "
                            "member in np.number")
        if self.pianoroll.ndim != 2:
            raise ValueError("Dimension of `pianoroll` must be 2")
        if self.pianoroll.shape[1] != 128:
            raise ValueError("Time axis length of `pianoroll` must be 128")
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
        remainder = self.pianoroll.shape[0] % factor
        if remainder:
            pad_width = ((0, (factor - remainder)), (0, 0))
            self.pianoroll = np.pad(self.pianoroll, pad_width, 'constant')

    def get_pianoroll_copy(self):
        """Return a copy of the piano-roll matrix."""
        return np.copy(self.pianoroll)

    def get_active_length(self):
        """
        Return the active length (i.e. without trailing silence) of the
        piano-roll (in time step).

        Returns
        -------
        active_length : int
            Length of the piano-roll without trailing silence (in time step).

        """
        non_zero_steps = np.any(self.pianoroll, axis=1)
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
        is_binarized = np.issubdtype(self.pianoroll.dtype, np.bool_)
        return is_binarized

    def plot(self, **kwargs):
        """Plot the piano-roll or save a plot of the piano-roll. See
        :func:`pypianoroll.plot.plot_track` for full documentation."""
        return plot_track(self, **kwargs)


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

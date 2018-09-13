"""Class for single-track pianorolls with metadata.

"""
from __future__ import absolute_import, division, print_function
from copy import deepcopy
from six import string_types
import numpy as np
from pypianoroll.plot import plot_track

class Track(object):
    """
    A single-track pianoroll container.

    Attributes
    ----------
    pianoroll : np.ndarray, shape=(num_time_step, 128)
        The pianoroll matrix. The first and second dimension represents time
        and pitch, respectively. Available data types are bool, int and float.
    program: int
        The program number according to General MIDI specification. Available
        values are 0 to 127.
    is_drum : bool
        The boolean indicator that indicates whether it is a percussion track.
    name : str
        The name of the track.

    """
    def __init__(self, pianoroll=None, program=0, is_drum=False,
                 name='unknown'):
        """
        Initialize the object by assigning attributes.

        Parameters
        ----------
        pianoroll : np.ndarray, shape=(num_time_step, 128)
            A pianoroll matrix. The first and second dimension represents time
            and pitch, respectively. Available datatypes are bool, int and
            float. Only effective when `track` is None.
        program: int
            A program number according to General MIDI specification [1].
            Available values are 0 to 127. Defaults to 0 (Acoustic Grand Piano).
        is_drum : bool
            A boolean number that indicates whether it is a percussion track.
            Defaults to False.
        name : str
            The name of the track. Defaults to 'unknown'.

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
        Assign a constant value to all nonzeros in the pianoroll. If the
        pianoroll is not binarized, its data type will be preserved. If the
        pianoroll is binarized, it will be casted to the type of `value`.

        Arguments
        ---------
        value : int or float
            The constant value to be assigned to all the nonzeros in the
            pianoroll.

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
        Binarize the pianoroll.

        Parameters
        ----------
        threshold : int or float
            A threshold used to binarize the pianorolls. Defaults to zero.

        """
        if not self.is_binarized():
            self.pianoroll = (self.pianoroll > threshold)

    def check_validity(self):
        """"Raise error if any invalid attribute found."""
        # pianoroll
        if not isinstance(self.pianoroll, np.ndarray):
            raise TypeError("`pianoroll` must be a numpy array.")
        if not (np.issubdtype(self.pianoroll.dtype, np.bool_)
                or np.issubdtype(self.pianoroll.dtype, np.number)):
            raise TypeError("The data type of `pianoroll` must be np.bool_ or "
                            "a subdtype of np.number.")
        if self.pianoroll.ndim != 2:
            raise ValueError("`pianoroll` must have exactly two dimensions.")
        if self.pianoroll.shape[1] != 128:
            raise ValueError("The length of the second axis of `pianoroll` "
                             "must be 128.")
        # program
        if not isinstance(self.program, int):
            raise TypeError("`program` must be int.")
        if self.program < 0 or self.program > 127:
            raise ValueError("`program` must be in between 0 to 127.")
        # is_drum
        if not isinstance(self.is_drum, bool):
            raise TypeError("`is_drum` must be bool.")
        # name
        if not isinstance(self.name, string_types):
            raise TypeError("`name` must be a string.")

    def clip(self, lower=0, upper=127):
        """
        Clip the pianoroll by the given lower and upper bounds.

        Parameters
        ----------
        lower : int or float
            The lower bound to clip the pianoroll. Defaults to 0.
        upper : int or float
            The upper bound to clip the pianoroll. Defaults to 127.

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

    def get_active_length(self):
        """
        Return the active length (i.e., without trailing silence) of the
        pianoroll. The unit is time step.

        Returns
        -------
        active_length : int
            The active length (i.e., without trailing silence) of the pianoroll.

        """
        nonzero_steps = np.any(self.pianoroll, axis=1)
        inv_last_nonzero_step = np.argmax(np.flip(nonzero_steps, axis=0))
        active_length = self.pianoroll.shape[0] - inv_last_nonzero_step
        return active_length

    def get_active_pitch_range(self):
        """
        Return the active pitch range as a tuple (lowest, highest).

        Returns
        -------
        lowest : int
            The lowest active pitch in the pianoroll.
        highest : int
            The highest active pitch in the pianoroll.

        """
        if self.pianoroll.shape[1] < 1:
            raise ValueError("Cannot compute the active pitch range for an "
                             "empty pianoroll")
        lowest = 0
        highest = 127
        while lowest < highest:
            if np.any(self.pianoroll[:, lowest]):
                break
            lowest += 1
        if lowest == highest:
            raise ValueError("Cannot compute the active pitch range for an "
                             "empty pianoroll")
        while not np.any(self.pianoroll[:, highest]):
            highest -= 1

        return lowest, highest

    def get_pianoroll_copy(self):
        """
        Return a copy of the pianoroll matrix.

        Returns
        -------
        copied : np.ndarray
            A copy of the pianoroll matrix.

        """
        copied = np.copy(self.pianoroll)
        return copied

    def is_binarized(self):
        """
        Return True if the pianoroll is already binarized. Otherwise, return
        False.

        Returns
        -------
        is_binarized : bool
            True if the pianoroll is already binarized; otherwise, False.

        """
        is_binarized = np.issubdtype(self.pianoroll.dtype, np.bool_)
        return is_binarized

    def pad(self, pad_length):
        """
        Pad the pianoroll with zeros at the end along the time axis.

        Parameters
        ----------
        pad_length : int
            The length to pad with zeros along the time axis.

        """
        self.pianoroll = np.pad(
            self.pianoroll, ((0, pad_length), (0, 0)), 'constant')

    def pad_to_multiple(self, factor):
        """
        Pad the pianoroll with zeros at the end along the time axis with the
        minimum length that makes the resulting pianoroll length a multiple of
        `factor`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting pianoroll will be
            a multiple of.

        """
        remainder = self.pianoroll.shape[0] % factor
        if remainder:
            pad_width = ((0, (factor - remainder)), (0, 0))
            self.pianoroll = np.pad(self.pianoroll, pad_width, 'constant')

    def plot(self, **kwargs):
        """Plot the pianoroll or save a plot of ot. See
        :func:`pypianoroll.plot.plot_track` for full documentation."""
        return plot_track(self, **kwargs)

    def transpose(self, semitone):
        """
        Transpose the pianoroll by a number of semitones, where positive
        values are for higher key, while negative values are for lower key.

        Parameters
        ----------
        semitone : int
            The number of semitones to transpose the pianoroll.

        """
        if semitone > 0 and semitone < 128:
            self.pianoroll[:, semitone:] = self.pianoroll[:, :(128 - semitone)]
            self.pianoroll[:, :semitone] = 0
        elif semitone < 0 and semitone > -128:
            self.pianoroll[:, :(128 + semitone)] = self.pianoroll[:, -semitone:]
            self.pianoroll[:, (128 + semitone):] = 0

    def trim_trailing_silence(self):
        """Trim the trailing silence of the pianoroll."""
        length = self.get_active_length()
        self.pianoroll = self.pianoroll[:length]

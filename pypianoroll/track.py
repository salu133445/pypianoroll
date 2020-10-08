"""Class for single-track piano rolls."""
from copy import deepcopy
from typing import Any, Optional

import numpy as np
from numpy import ndarray

from .visualization import plot_track

__all__ = ["Track"]


class Track:
    """
    A container for single-track piano roll.

    Attributes
    ----------
    program : int, 0-127, optional
        Program number according to General MIDI specification [1]. Defaults
        to 0 (Acoustic Grand Piano).
    is_drum : bool, optional
        Whether it is a percussion track. Defaults to False.
    name : str, optional
        Track name.
    pianoroll : ndarray, dtype={bool, int}, shape=(?, 128), optional
        Piano-roll matrix. The first dimension represents time, and the
        second dimension represents pitch. If dtype is integer, assume the
        data range is in [0, 255].

    References
    ----------
    [1] https://www.midi.org/specifications/item/gm-level-1-sound-set

    """

    def __init__(
        self,
        program: int = 0,
        is_drum: bool = False,
        name: Optional[str] = None,
        pianoroll: Optional[ndarray] = None,
    ):
        self.program = program
        self.is_drum = is_drum
        self.name = name
        self.pianoroll = np.array([]) if pianoroll is None else pianoroll

    def __getitem__(self, val):
        return Track(
            program=self.program,
            is_drum=self.is_drum,
            name=self.name,
            pianoroll=self.pianoroll[val],
        )

    def __repr__(self):
        to_join = []
        for attr in ("program", "is_drum", "name"):
            value = getattr(self, attr)
            if value is not None:
                to_join.append(attr + "=" + repr(value))
        to_join.append(
            "pianoroll=array(shape={}, dtype={})".format(
                self.pianoroll.shape, self.pianoroll.dtype
            )
        )
        return "Track(" + ", ".join(to_join) + ")"

    def validate(self):
        """Raise a proper error if any attribute is invalid."""
        if not isinstance(self.program, int):
            raise TypeError("`program` must be of type int.")
        if self.program < 0 or self.program > 127:
            raise ValueError("`program` must be in between 0 to 127.")
        if not isinstance(self.is_drum, bool):
            raise TypeError("`is_drum` must be of type bool.")
        if not isinstance(self.name, str):
            raise TypeError("`name` must be of type str.")
        if not isinstance(self.pianoroll, ndarray):
            raise TypeError("`pianoroll` must be an ndarray.")
        if not (
            np.issubdtype(self.pianoroll.dtype, np.bool_)
            or np.issubdtype(self.pianoroll.dtype, np.number)
        ):
            raise TypeError(
                "The data type of `pianoroll` must be np.bool_ or a subdtype "
                "of np.number."
            )
        if self.pianoroll.ndim != 2:
            raise ValueError("`pianoroll` must have exactly two dimensions.")
        if self.pianoroll.shape[1] != 128:
            raise ValueError(
                "The length of the second axis of `pianoroll` must be 128."
            )

    def is_valid(self):
        """Return True if all attributes are valid, otherwise False.

        Returns
        -------
        bool
            Whether all attributes is valid.

        """
        try:
            self.validate()
        except (TypeError, ValueError):
            return False
        return True

    def assign_constant(self, value: float, dtype: Any = None):
        """Assign a constant value to all nonzeros entries of the piano roll.

        If the piano roll is not binarized, its data type will be preserved. If
        the piano roll is binarized, cast it to the dtype of `value`.

        Arguments
        ---------
        value : int or float
            Value to assign to all the nonzero entries in the piano roll.

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

    def binarize(self, threshold: float = 0):
        """
        Binarize the piano roll.

        Parameters
        ----------
        threshold : int or float
            A threshold used to binarize the piano roll. Defaults to zero.

        """
        if not self.is_binarized():
            self.pianoroll = self.pianoroll > threshold

    def clip(self, lower: float = 0, upper: float = 127):
        """Clip the piano roll by a lower bound and an upper bound.

        Parameters
        ----------
        lower : int or float
            Lower bound to clip the piano roll. Defaults to 0.
        upper : int or float
            Upper bound to clip the piano roll. Defaults to 127.

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
        """Return the active length of the piano roll (in time steps).

        The active length is defined as the length of the piano roll without
        trailing silence.

        Returns
        -------
        int
            Active length.

        """
        nonzero_steps = np.any(self.pianoroll, axis=1)
        inv_last_nonzero_step = np.argmax(np.flip(nonzero_steps, axis=0))
        active_length = self.pianoroll.shape[0] - inv_last_nonzero_step
        return active_length

    def get_active_pitch_range(self):
        """Return the active pitch range as a tuple (lowest, highest).

        Returns
        -------
        lowest : int
            Lowest active pitch in the piano roll.
        highest : int
            Highest active pitch in the piano roll.

        """
        if self.pianoroll.shape[1] < 1:
            raise ValueError(
                "Cannot compute the active pitch range for an empty piano roll"
            )
        lowest = 0
        highest = 127
        while lowest < highest:
            if np.any(self.pianoroll[:, lowest]):
                break
            lowest += 1
        if lowest == highest:
            raise ValueError(
                "Cannot compute the active pitch range for an empty piano roll"
            )
        while not np.any(self.pianoroll[:, highest]):
            highest -= 1

        return lowest, highest

    def get_pianoroll_copy(self):
        """Return a copy of the piano roll matrix.

        Returns
        -------
        copied : ndarray
            A copy of the pianoroll matrix.

        """
        copied = np.copy(self.pianoroll)
        return copied

    def is_binarized(self):
        """Return True if the piano roll is binarized, otherwise return False.

        Returns
        -------
        bool
            Whether the piano roll is binarized.

        """
        return np.issubdtype(self.pianoroll.dtype, np.bool_)

    def pad(self, pad_length: int):
        """Pad the piano roll with zeros at the end along the time axis.

        Parameters
        ----------
        pad_length : int
            The length to pad with zeros along the time axis.

        """
        self.pianoroll = np.pad(
            self.pianoroll, ((0, pad_length), (0, 0)), "constant"
        )

    def pad_to_multiple(self, factor: int):
        """Pad the piano roll along the time axis to a multiple of `factor`.

        Pad the piano roll with zeros at the end along the time axis of the
        minimum length that makes the length of the resulting piano roll a
        multiple of `factor`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting piano roll will be
            a multiple of.

        """
        remainder = self.pianoroll.shape[0] % factor
        if remainder:
            pad_width = ((0, (factor - remainder)), (0, 0))
            self.pianoroll = np.pad(self.pianoroll, pad_width, "constant")

    def plot(self, **kwargs):
        """Plot the piano roll and/or save a plot of it.

        See :func:`pypianoroll.plot_track` for full documentation.

        """
        return plot_track(self, **kwargs)

    def transpose(self, semitone: int):
        """Transpose the piano roll by a number of semitones.

        Positive values are for a higher key, while negative values are for
        a lower key.

        Parameters
        ----------
        semitone : int
            Number of semitones to transpose the piano roll.

        """
        if 0 < semitone < 128:
            self.pianoroll[:, semitone:] = self.pianoroll[
                :, : (128 - semitone)
            ]
            self.pianoroll[:, :semitone] = 0
        elif -128 < semitone < 0:
            self.pianoroll[:, : (128 + semitone)] = self.pianoroll[
                :, -semitone:
            ]
            self.pianoroll[:, (128 + semitone) :] = 0

    def trim_trailing_silence(self):
        """Trim the trailing silence of the piano roll."""
        length = self.get_active_length()
        self.pianoroll = self.pianoroll[:length]

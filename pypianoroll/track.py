"""Classes for single-track piano rolls.

Classes
-------

- BinaryTrack
- StandardTrack
- Track

Variables
---------

- DEFAULT_PROGRAM
- DEFAULT_IS_DRUM

"""
from typing import Optional, Tuple, TypeVar

import numpy as np
from numpy import ndarray

from .visualization import plot_track

__all__ = [
    "BinaryTrack",
    "StandardTrack",
    "Track",
    "DEFAULT_PROGRAM",
    "DEFAULT_IS_DRUM",
]

DEFAULT_PROGRAM = 0
DEFAULT_IS_DRUM = False

T = TypeVar("T", bound="Track")
ST = TypeVar("ST", bound="StandardTrack")


class Track:
    """A generic container for single-track piano rolls.

    Attributes
    ----------
    program : int, 0-127, optional
        Program number according to General MIDI specification [1].
        Defaults to 0 (Acoustic Grand Piano).
    is_drum : bool, optional
        Whether it is a percussion track. Defaults to False.
    name : str, optional
        Track name.
    pianoroll : ndarray, shape=(?, 128), optional
        Piano-roll matrix. The first dimension represents time, and the
        second dimension represents pitch.

    References
    ----------
    1. https://www.midi.org/specifications/item/gm-level-1-sound-set

    """

    def __init__(
        self,
        name: Optional[str] = None,
        program: Optional[int] = None,
        is_drum: Optional[bool] = None,
        pianoroll: Optional[ndarray] = None,
    ):
        self.name = name
        self.program = program if program is not None else DEFAULT_PROGRAM
        self.is_drum = is_drum if is_drum is not None else DEFAULT_IS_DRUM
        if pianoroll is None:
            self.pianoroll = np.array([])
        else:
            self.pianoroll = np.asarray(pianoroll)

    def __repr__(self) -> str:
        to_join = [
            f"name={repr(self.name)}",
            f"program={repr(self.program)}",
            f"is_drum={repr(self.is_drum)}",
            f"pianoroll=array(shape={self.pianoroll.shape})",
        ]
        return f"Track({', '.join(to_join)})"

    def _validate_type(self, attr):
        if getattr(self, attr) is None:
            if attr in ("program", "is_drum"):
                raise TypeError(f"`{attr}` must not be None.")
            return
        if attr == "program":
            if not isinstance(self.program, int):
                raise TypeError(
                    "`program` must be of type int, not "
                    f"{type(self.program)}."
                )
        elif attr == "is_drum":
            if not isinstance(self.is_drum, bool):
                raise TypeError(
                    "`is_drum` must be of type bool, not "
                    f"{type(self.is_drum)}."
                )
        elif attr == "name":
            if not isinstance(self.name, str):
                raise TypeError(
                    f"`name` must be of type str, not {type(self.name)}."
                )
        elif attr == "pianoroll":
            if not isinstance(self.pianoroll, ndarray):
                raise TypeError(
                    "`pianoroll` must be a NumPy array, not "
                    f"{type(self.pianoroll)}."
                )

    def validate_type(self, attr=None):
        """Raise an error if an attribute has an invalid type.

        Parameters
        ----------
        attr : str
            Attribute to validate. Defaults to validate all attributes.

        Returns
        -------
        Object itself.

        """
        if attr is None:
            for attribute in ("program", "is_drum", "name", "pianoroll"):
                self._validate_type(attribute)
        else:
            self._validate_type(attr)
        return self

    def _validate(self, attr):
        self._validate_type(attr)
        if attr == "program":
            if self.program < 0 or self.program > 127:
                raise ValueError("`program` must be in between 0 to 127.")
        elif attr == "pianoroll":
            if self.pianoroll.ndim != 2:
                raise ValueError(
                    "`pianoroll` must have exactly two dimensions."
                )
            if self.pianoroll.shape[1] != 128:
                raise ValueError(
                    "Length of the second axis of `pianoroll` must be 128."
                )

    def validate(self, attr=None):
        """Raise an error if an attribute has an invalid type or value.

        Parameters
        ----------
        attr : str
            Attribute to validate. Defaults to validate all attributes.

        Returns
        -------
        Object itself.

        """
        if attr is None:
            for attribute in ("program", "is_drum", "name", "pianoroll"):
                self._validate(attribute)
        else:
            self._validate(attr)
        return self

    def is_valid_type(self, attr: Optional[str] = None) -> bool:
        """Return True if an attribute is of a valid type.

        Parameters
        ----------
        attr : str
            Attribute to validate. Defaults to validate all attributes.

        Returns
        -------
        bool
            Whether the attribute is of a valid type.

        """
        try:
            self.validate_type(attr)
        except TypeError:
            return False
        return True

    def is_valid(self, attr: Optional[str] = None) -> bool:
        """Return True if an attribute is valid.

        Parameters
        ----------
        attr : str
            Attribute to validate. Defaults to validate all attributes.

        Returns
        -------
        bool
            Whether the attribute has a valid type and value.

        """
        try:
            self.validate(attr)
        except (TypeError, ValueError):
            return False
        return True

    def get_active_length(self) -> int:
        """Return the active length of the piano roll.

        Returns
        -------
        int
            Length (in time steps) of the piano roll without trailing
            silence.

        """
        nonzero_steps = np.any(self.pianoroll, axis=1)
        inv_last_nonzero_step = np.argmax(np.flip(nonzero_steps, axis=0))
        active_length = self.pianoroll.shape[0] - inv_last_nonzero_step
        return active_length

    def get_active_pitch_range(self) -> Tuple[int, int]:
        """Return the active pitch range as a tuple (lowest, highest).

        Returns
        -------
        int
            Lowest active pitch in the piano roll.
        int
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

    def pad(self: T, pad_length: int) -> T:
        """Pad the piano roll.

        Parameters
        ----------
        pad_length : int
            Length to pad along the time axis.

        Returns
        -------
        Object itself.

        See Also
        --------
        :func:`pypianoroll.Track.pad_to_multiple` : Pad the piano
          roll so that its length is some multiple.

        """
        self.pianoroll = np.pad(
            self.pianoroll, ((0, pad_length), (0, 0)), "constant"
        )
        return self

    def pad_to_multiple(self: T, factor: int) -> T:
        """Pad the piano roll so that its length is some multiple.

        Pad the piano roll at the end along the time axis of
        the minimum length that makes the length of the resulting piano
        roll a multiple of `factor`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting piano roll will
            be a multiple of.

        Returns
        -------
        Object itself.

        See Also
        --------
        :func:`pypianoroll.Track.pad` : Pad the piano roll.

        """
        remainder = self.pianoroll.shape[0] % factor
        if remainder:
            pad_width = ((0, (factor - remainder)), (0, 0))
            self.pianoroll = np.pad(self.pianoroll, pad_width, "constant")
        return self

    def transpose(self: T, semitone: int) -> T:
        """Transpose the piano roll by a number of semitones.

        Parameters
        ----------
        semitone : int
            Number of semitones to transpose. A positive value raises
            the pitches, while a negative value lowers the pitches.

        Returns
        -------
        Object itself.

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
        return self

    def trim_trailing_silence(self: T) -> T:
        """Trim the trailing silence of the piano roll.

        Returns
        -------
        Object itself.

        """
        length = self.get_active_length()
        self.pianoroll = self.pianoroll[:length]
        return self

    def plot(self, **kwargs):
        """Plot the piano roll.

        Refer to :func:`pypianoroll.plot_track` for full documentation.

        """
        return plot_track(self, **kwargs)


class StandardTrack(Track):
    """A container for single-track piano rolls with velocities.

    Attributes
    ----------
    program : int, 0-127, optional
        Program number according to General MIDI specification [1].
        Defaults to 0 (Acoustic Grand Piano).
    is_drum : bool, optional
        Whether it is a percussion track. Defaults to False.
    name : str, optional
        Track name.
    pianoroll : ndarray, dtype=uint8, shape=(?, 128), optional
        Piano-roll matrix. The first dimension represents time, and the
        second dimension represents pitch. Cast to uint8 if not of data
        type uint8.

    References
    ----------
    1. https://www.midi.org/specifications/item/gm-level-1-sound-set

    """

    def __init__(
        self,
        program: int = 0,
        is_drum: bool = False,
        name: Optional[str] = None,
        pianoroll: Optional[ndarray] = None,
    ):
        super().__init__(name, program, is_drum, pianoroll)
        self.pianoroll = self.pianoroll.astype(np.uint8)

    def __repr__(self):
        to_join = [
            f"name={repr(self.name)}",
            f"program={repr(self.program)}",
            f"is_drum={repr(self.is_drum)}",
            f"pianoroll=array(shape={self.pianoroll.shape})",
        ]
        return f"StandardTrack({', '.join(to_join)})"

    def _validate_type(self, attr):
        super()._validate_type(attr)
        if attr == "pianoroll" and self.pianoroll.dtype != np.uint8:
            raise TypeError(
                "`pianoroll` must be of data type uint8, not "
                f"{self.pianoroll.dtype}."
            )

    def _validate(self, attr):
        super()._validate(attr)
        if attr == "pianoroll" and np.any(self.pianoroll > 127):
            raise ValueError(
                "`pianoroll` must contain only integers between 0 to 127."
            )

    def assign_constant(self: ST, value: int) -> ST:
        """Assign a constant value to all nonzeros entries.

        Arguments
        ---------
        value : int
            Value to assign.

        Returns
        -------
        Object itself.

        """
        self.pianoroll[self.pianoroll.nonzero()] = value
        return self

    def binarize(self, threshold: float = 0) -> "BinaryTrack":
        """Binarize the piano roll.

        Parameters
        ----------
        threshold : int or float
            Threshold for binarizing the piano roll. Defaults to 0.

        Returns
        -------
        Object itself.

        """
        return BinaryTrack(
            program=self.program,
            is_drum=self.is_drum,
            name=self.name,
            pianoroll=(self.pianoroll > threshold),
        )

    def clip(self: ST, lower: int = 0, upper: int = 127) -> ST:
        """Clip (limit) the the piano roll into [`lower`, `upper`].

        Parameters
        ----------
        lower : int
            Lower bound. Defaults to 0.
        upper : int
            Upper bound. Defaults to 127.

        Returns
        -------
        Object itself.

        """
        if not isinstance(lower, int):
            raise ValueError("`lower` must be of type int.")
        if not isinstance(upper, int):
            raise ValueError("`upper` must be of type int.")
        self.pianoroll = self.pianoroll.clip(lower, upper)
        return self


class BinaryTrack(Track):
    """A container for single-track, binary piano rolls.

    Attributes
    ----------
    program : int, 0-127, optional
        Program number according to General MIDI specification [1].
        Defaults to 0 (Acoustic Grand Piano).
    is_drum : bool, optional
        Whether it is a percussion track. Defaults to False.
    name : str, optional
        Track name.
    pianoroll : ndarray, dtype=bool, shape=(?, 128), optional
        Piano-roll matrix. The first dimension represents time, and the
        second dimension represents pitch. Cast to bool if not of data
        type uint8.

    References
    ----------
    1. https://www.midi.org/specifications/item/gm-level-1-sound-set

    """

    def __init__(
        self,
        program: int = 0,
        is_drum: bool = False,
        name: Optional[str] = None,
        pianoroll: Optional[ndarray] = None,
    ):
        super().__init__(name, program, is_drum, pianoroll)
        self.pianoroll = self.pianoroll.astype(np.bool_)

    def __repr__(self):
        to_join = [
            f"name={repr(self.name)}",
            f"program={repr(self.program)}",
            f"is_drum={repr(self.is_drum)}",
            f"pianoroll=array(shape={self.pianoroll.shape})",
        ]
        return f"BinaryTrack({', '.join(to_join)})"

    def _validate_type(self, attr):
        super()._validate_type(attr)
        if attr == "pianoroll" and self.pianoroll.dtype != np.bool_:
            raise TypeError(
                "`pianoroll` must be of data type bool, not "
                f"{self.pianoroll.dtype}."
            )

    def assign_constant(self, value: int) -> "Track":
        """Assign a constant value to all nonzeros entries.

        Arguments
        ---------
        value : int
            Value to assign.

        Returns
        -------
        Object itself.

        """
        pianoroll = np.zeros(self.pianoroll.shape, np.uint8)
        pianoroll[self.pianoroll.nonzero()] = value
        return Track(
            program=self.program,
            is_drum=self.is_drum,
            name=self.name,
            pianoroll=pianoroll,
        )

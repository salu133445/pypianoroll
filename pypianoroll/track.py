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
from typing import Optional, TypeVar

import numpy as np
from matplotlib.axes import Axes
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

_Track = TypeVar("_Track", bound="Track")
_StandardTrack = TypeVar("_StandardTrack", bound="StandardTrack")


class Track:
    """A generic container for single-track piano rolls.

    Attributes
    ----------
    name : str, optional
        Track name.
    program : int, 0-127, optional
        Program number according to General MIDI specification [1].
        Defaults to 0 (Acoustic Grand Piano).
    is_drum : bool, optional
        Whether it is a percussion track. Defaults to False.
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
            self.pianoroll = np.zeros((0, 128))
        else:
            self.pianoroll = np.asarray(pianoroll)

    def __repr__(self) -> str:
        to_join = [
            f"name={repr(self.name)}",
            f"program={repr(self.program)}",
            f"is_drum={repr(self.is_drum)}",
            f"pianoroll=array(shape={self.pianoroll.shape}, "
            f"dtype={self.pianoroll.dtype})",
        ]
        return f"Track({', '.join(to_join)})"

    def __len__(self) -> int:
        return len(self.pianoroll)

    def __getitem__(self, key) -> ndarray:
        return self.pianoroll[key]

    def _validate_type(self, attr):
        if getattr(self, attr) is None:
            if attr in ("program", "is_drum", "pianoroll"):
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
        if getattr(self, attr) is None:
            if attr in ("program", "is_drum", "pianoroll"):
                raise TypeError(f"`{attr}` must not be None.")
            return
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

    def get_length(self) -> int:
        """Return the active length of the piano roll.

        Returns
        -------
        int
            Length (in time steps) of the piano roll without trailing
            silence.

        """
        nonzero_steps = np.any(self.pianoroll, axis=1)
        inv_last_nonzero_step = np.argmax(np.flip(nonzero_steps, axis=0))
        return self.pianoroll.shape[0] - inv_last_nonzero_step

    def copy(self):
        """Return a copy of the track.

        Returns
        -------
        A copy of the object itself.

        Notes
        -----
        The piano-roll array is copied using :func:`numpy.copy`.

        """
        return Track(
            name=self.name,
            program=self.program,
            is_drum=self.is_drum,
            pianoroll=self.pianoroll.copy(),
        )

    def pad(self: _Track, pad_length: int) -> _Track:
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

    def pad_to_multiple(self: _Track, factor: int) -> _Track:
        """Pad the piano roll so that its length is some multiple.

        Pad the piano roll at the end along the time axis of the minimum
        length that makes the length of the resulting piano roll a
        multiple of `factor`.

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

    def transpose(self: _Track, semitone: int) -> _Track:
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

    def trim(
        self: _Track, start: Optional[int] = None, end: Optional[int] = None
    ) -> _Track:
        """Trim the piano roll.

        Parameters
        ----------
        start : int, optional
            Start time. Defaults to 0.
        end : int, optional
            End time. Defaults to active length.

        Returns
        -------
        Object itself.

        """
        if start is None:
            start = 0
        elif start < 0:
            raise ValueError("`start` must be nonnegative.")
        if end is None:
            end = self.get_length()
        elif end > len(self.pianoroll):
            raise ValueError(
                "`end` must be shorter than the piano roll length."
            )
        self.pianoroll = self.pianoroll[start:end]
        return self

    def standardize(self: "Track") -> "StandardTrack":
        """Standardize the track.

        This will clip the piano roll to [0, 127] and cast to np.uint8.

        Returns
        -------
        Converted StandardTrack object.

        """
        return StandardTrack(
            name=self.name,
            program=self.program,
            is_drum=self.is_drum,
            pianoroll=np.clip(self.pianoroll, 0, 127),
        )

    def binarize(self, threshold: float = 0) -> "BinaryTrack":
        """Binarize the track.

        This will binarize the piano roll by the given threshold.

        Parameters
        ----------
        threshold : int or float
            Threshold. Defaults to 0.

        Returns
        -------
        Converted BinaryTrack object.

        """
        return BinaryTrack(
            program=self.program,
            is_drum=self.is_drum,
            name=self.name,
            pianoroll=(self.pianoroll > threshold),
        )

    def plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """Plot the piano roll.

        Refer to :func:`pypianoroll.plot_track` for full documentation.

        """
        return plot_track(self, ax, **kwargs)


class StandardTrack(Track):
    """A container for single-track piano rolls with velocities.

    Attributes
    ----------
    name : str, optional
        Track name.
    program : int, 0-127, optional
        Program number according to General MIDI specification [1].
        Defaults to 0 (Acoustic Grand Piano).
    is_drum : bool, optional
        Whether it is a percussion track. Defaults to False.
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
        name: Optional[str] = None,
        program: Optional[int] = None,
        is_drum: Optional[bool] = None,
        pianoroll: Optional[ndarray] = None,
    ):
        super().__init__(name, program, is_drum, pianoroll)
        if self.pianoroll.dtype != np.uint8:
            self.pianoroll = self.pianoroll.astype(np.uint8)

    def __repr__(self):
        to_join = [
            f"name={repr(self.name)}",
            f"program={repr(self.program)}",
            f"is_drum={repr(self.is_drum)}",
            f"pianoroll=array(shape={self.pianoroll.shape}, "
            f"dtype={self.pianoroll.dtype})",
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

    def set_nonzeros(self: _StandardTrack, value: int) -> _StandardTrack:
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

    def clip(
        self: _StandardTrack, lower: int = 0, upper: int = 127
    ) -> _StandardTrack:
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
    name : str, optional
        Track name.
    program : int, 0-127, optional
        Program number according to General MIDI specification [1].
        Defaults to 0 (Acoustic Grand Piano).
    is_drum : bool, optional
        Whether it is a percussion track. Defaults to False.
    pianoroll : ndarray, dtype=bool, shape=(?, 128), optional
        Piano-roll matrix. The first dimension represents time, and the
        second dimension represents pitch. Cast to bool if not of data
        type bool.

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
        super().__init__(name, program, is_drum, pianoroll)
        if self.pianoroll.dtype != np.bool_:
            self.pianoroll = self.pianoroll.astype(np.bool_)

    def __repr__(self):
        to_join = [
            f"name={repr(self.name)}",
            f"program={repr(self.program)}",
            f"is_drum={repr(self.is_drum)}",
            f"pianoroll=array(shape={self.pianoroll.shape}, "
            f"dtype={self.pianoroll.dtype})",
        ]
        return f"BinaryTrack({', '.join(to_join)})"

    def _validate_type(self, attr):
        super()._validate_type(attr)
        if attr == "pianoroll" and self.pianoroll.dtype != np.bool_:
            raise TypeError(
                "`pianoroll` must be of data type bool, not "
                f"{self.pianoroll.dtype}."
            )

    def set_nonzeros(self, value: int) -> "StandardTrack":
        """Assign a constant value to all nonzeros entries.

        Arguments
        ---------
        value : int
            Value to assign.

        Returns
        -------
        Converted StandardTrack object.

        """
        pianoroll = np.zeros(self.pianoroll.shape, np.uint8)
        pianoroll[self.pianoroll.nonzero()] = value
        return StandardTrack(
            name=self.name,
            program=self.program,
            is_drum=self.is_drum,
            pianoroll=pianoroll,
        )

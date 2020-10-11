"""Multitrack class.

This module defines the core class of Pypianoroll---the Multitrack
class, a container for multitrack piano rolls.

Classes
-------

- Multitrack

Variables
---------

- DEFAULT_RESOLUTION
- DEFAULT_TEMPO

"""
from typing import List, Optional, Tuple, TypeVar

import numpy as np
from numpy import ndarray

from .outputs import save, to_pretty_midi, write
from .track import BinaryTrack, StandardTrack, Track
from .visualization import plot_multitrack

__all__ = [
    "Multitrack",
    "DEFAULT_RESOLUTION",
    "DEFAULT_TEMPO",
]

DEFAULT_RESOLUTION = 24
DEFAULT_TEMPO = 120

M = TypeVar("M", bound="Multitrack")


class Multitrack:
    """A container for multitrack piano rolls.

    This is the core class of Pypianoroll.

    Attributes
    ----------
    name : str, optional
        Multitrack name.
    resolution : int
        Time steps per quarter note.
    tempo : ndarray, dtype=float, shape=(?, 1), optional
        Tempo (in qpm) at each time step. Length is the total number
        of time steps. Cast to float if not of data type float.
    downbeat : ndarray, dtype=bool, shape=(?, 1), optional
        Boolean array that indicates whether the time step contains a
        downbeat (i.e., the first time step of a bar). Length is the
        total number of time steps.
    tracks : list of :class:`pypianoroll.Track`, optional
        Music tracks.

    """

    def __init__(
        self,
        name: Optional[str] = None,
        resolution: Optional[int] = None,
        tempo: Optional[ndarray] = None,
        downbeat: Optional[ndarray] = None,
        tracks: Optional[List[Track]] = None,
    ):
        self.name = name

        if resolution is not None:
            self.resolution = resolution
        else:
            self.resolution = DEFAULT_RESOLUTION

        if tempo is None:
            self.tempo = None
        elif np.issubdtype(tempo.dtype, np.floating):
            self.tempo = tempo
        else:
            self.tempo = np.asarray(tempo).astype(float)

        if downbeat is None:
            self.downbeat = None
        elif np.issubdtype(downbeat.dtype, np.bool_):
            self.downbeat = downbeat
        else:
            self.downbeat = np.asarray(downbeat).astype(bool)

        self.tracks = tracks if tracks is not None else []

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, val):
        if isinstance(val, tuple):
            if isinstance(val[0], int):
                tracks = [self.tracks[val[0]][val[1:]]]
            elif isinstance(val[0], list):
                tracks = [self.tracks[i][val[1:]] for i in val[0]]
            else:
                tracks = [track[val[1:]] for track in self.tracks[val[0]]]

            if self.downbeat is not None:
                downbeat = self.downbeat[val[1]]
            else:
                downbeat = None

            if self.tempo is not None:
                tempo = self.tempo[val[1]]
            else:
                tempo = None

            return Multitrack(
                name=self.name,
                resolution=self.resolution,
                tempo=tempo,
                downbeat=downbeat,
                tracks=tracks,
            )

        if isinstance(val, list):
            tracks = [self.tracks[i] for i in val]
        else:
            tracks = self.tracks[val]

        return Multitrack(
            name=self.name,
            resolution=self.resolution,
            tempo=self.tempo,
            downbeat=self.downbeat,
            tracks=tracks,
        )

    def __repr__(self) -> str:
        if self.tempo is None:
            tempo_repr = "None"
        else:
            tempo_repr = f"array(shape={self.tempo.shape})"

        if self.downbeat is None:
            downbeat_repr = "None"
        else:
            downbeat_repr = f"array(shape={self.downbeat.shape})"

        to_join = [
            f"name={repr(self.name)}",
            f"resolution={repr(self.resolution)}",
            f"tempo={tempo_repr}",
            f"downbeat={downbeat_repr}",
            f"tracks={repr(self.tracks)}",
        ]
        return f"Multitrack({', '.join(to_join)})"

    def _validate_type(self, attr):
        if getattr(self, attr) is None:
            if attr == "resolution":
                raise TypeError(f"`{attr}` must not be None.")
            return

        if attr == "name":
            if not isinstance(self.name, str):
                raise TypeError(
                    "`name` must be of type str, but got type "
                    f"{type(self.name)}."
                )

        elif attr == "resolution":
            if not isinstance(self.resolution, int):
                raise TypeError(
                    "`resolution` must be of type int, but got "
                    f"{type(self.resolution)}."
                )
        elif attr == "tempo":
            if not isinstance(self.tempo, np.ndarray):
                raise TypeError("`tempo` must be a NumPy array.")
            if not np.issubdtype(self.tempo.dtype, np.number):
                raise TypeError(
                    "`tempo` must be of data type numpy.number, but got data "
                    f"type {self.tempo.dtype}."
                )
        elif attr == "downbeat":
            if not isinstance(self.downbeat, np.ndarray):
                raise TypeError("`downbeat` must be a NumPy array.")
            if not np.issubdtype(self.downbeat.dtype, np.bool_):
                raise TypeError(
                    "`downbeat` must be of data type bool, but got data type"
                    f"{self.downbeat.dtype}."
                )

        elif attr == "tracks":
            for i, track in enumerate(self.tracks):
                if not isinstance(track, Track):
                    raise TypeError(
                        "`tracks` must be a list of type Track, but got type "
                        f"{type(track)} at index {i}."
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
            attributes = ("name", "resolution", "tempo", "downbeat", "tracks")
            for attribute in attributes:
                self._validate_type(attribute)
        else:
            self._validate_type(attr)
        return self

    def _validate(self, attr):
        self._validate_type(attr)
        if attr == "resolution":
            if self.resolution < 1:
                raise ValueError("`resolution` must be a positive integer.")

        elif attr == "tempo":
            if self.tempo.ndim != 1:
                raise ValueError("`tempo` must be a 1D NumPy array.")
            if np.any(self.tempo <= 0.0):
                raise ValueError("`tempo` must contain only positive numbers.")

        elif attr == "downbeat":
            if self.downbeat.ndim != 1:
                raise ValueError("`downbeat` must be a 1D NumPy array.")

        elif attr == "tracks":
            for track in self.tracks:
                track.validate()

    def validate(self: M, attr=None) -> M:
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
            attributes = ("name", "resolution", "tempo", "downbeat", "tracks")
            for attribute in attributes:
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
        """Return the maximum active length of the piano rolls.

        Returns
        -------
        int
            Maximum active length (in time steps) of the piano rolls,
            where active length is the length of the piano roll without
            trailing silence.

        """
        active_length = 0
        for track in self.tracks:
            now_length = track.get_active_length()
            if active_length < track.get_active_length():
                active_length = now_length
        return active_length

    def get_active_pitch_range(self) -> Tuple[int, int]:
        """Return the active pitch range as a tuple (lowest, highest).

        Returns
        -------
        int
            Lowest active pitch in all the piano rolls.
        int
            Highest active pitch in all the piano rolls.

        """
        lowest, highest = self.tracks[0].get_active_pitch_range()
        if len(self.tracks) > 1:
            for track in self.tracks[1:]:
                low, high = track.get_active_pitch_range()
                if low < lowest:
                    lowest = low
                if high > highest:
                    highest = high
        return lowest, highest

    def get_downbeat_steps(self) -> ndarray:
        """Return the indices of time steps that contain downbeats.

        Returns
        -------
        ndarray, dtype=int
            Indices of time steps that contain downbeats.

        """
        if self.downbeat is None:
            return []
        return np.nonzero(self.downbeat)[0]

    def get_max_length(self) -> int:
        """Return the maximum length of the piano rolls.

        Returns
        -------
        int
            Maximum length (in time steps) of the piano rolls.

        """
        max_length = 0
        for track in self.tracks:
            if max_length < track.pianoroll.shape[0]:
                max_length = track.pianoroll.shape[0]
        return max_length

    def count_downbeat(self) -> int:
        """Return the number of down beats.

        Returns
        -------
        int
            Number of down beats.

        Note
        ----
        Return value is calculated based only on the attribute
        `downbeat`.

        """
        return np.count_nonzero(self.downbeat)

    def stack(self) -> ndarray:
        """Return the piano rolls stacked as a 3D tensor.

        Returns
        -------
        ndarray, shape=(?, ?, 128)
            Stacked piano roll, provided as *(track, time, pitch)*.

        """
        pianorolls = []
        max_length = self.get_max_length()
        for track in self.tracks:
            if track.pianoroll.shape[0] < max_length:
                pad_length = max_length - track.pianoroll.shape[0]
                padded = np.pad(
                    track.pianoroll, ((0, pad_length), (0, 0)), "constant",
                )
                pianorolls.append(padded)
            else:
                pianorolls.append(track.pianoroll)
        return np.stack(pianorolls)

    def blend(self, mode: str = "sum") -> ndarray:
        """Return the blended pianoroll.

        Parameters
        ----------
        mode : {'sum', 'max', 'any'}
            Blending strategy to apply along the track axis.
            Defaults to 'sum'.

            'sum' (default)
                Sum the piano rolls. Note that for binary piano rolls, integer
                summation is performed.
            'max'
                for each pixel, the maximum value among
                all the piano rolls is assigned to the merged piano roll.
            'any'
                the value of a pixel in the merged piano
                roll is True if any of the piano rolls has nonzero value
                at that pixel; False if all piano rolls are inactive
                (zero-valued) at that pixel.

        Returns
        -------
        ndarray, shape=(?, 128)
            Blended piano roll.

        """
        stacked = self.stack()
        if mode.lower() == "any":
            return np.any(stacked, axis=0)
        if mode.lower() == "sum":
            return np.sum(stacked, axis=0).clip(0, 127).astype(np.uint8)
        if mode.lower() == "max":
            return np.max(stacked, axis=0)
        raise ValueError("`mode` must be one of 'max', 'sum' and 'any'.")

    def append(self: M, track: Track) -> M:
        """Append a Track object to the track list.

        Parameters
        ----------
        track : :class:`pypianoroll.Track`
            Track to append.

        Returns
        -------
        Object itself.

        """
        self.tracks.append(track)
        return self

    def remove_empty(self: M) -> M:
        """Remove tracks with empty pianorolls."""
        self.tracks = [
            track for track in self.tracks if not np.any(track.pianoroll)
        ]
        return self

    def assign_constant(self: M, value: int) -> M:
        """Assign a constant value to all nonzero entries.

        Arguments
        ---------
        value : int
            Value to assign.

        Returns
        -------
        Object itself.

        """
        for i, track in enumerate(self.tracks):
            if isinstance(track, StandardTrack):
                track.assign_constant(value)
            elif isinstance(track, BinaryTrack):
                self.tracks[i] = track.assign_constant(value)
        return self

    def binarize(self: M, threshold: float = 0) -> M:
        """Binarize the piano rolls.

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano rolls. Defaults to zero.

        Returns
        -------
        Object itself.

        """
        for i, track in enumerate(self.tracks):
            if isinstance(track, StandardTrack):
                self.tracks[i] = track.binarize(threshold)
        return self

    def clip(self: M, lower: int = 0, upper: int = 127) -> M:
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
        for track in self.tracks:
            if isinstance(track, StandardTrack):
                track.clip(lower, upper)
        return self

    def downsample(self: M, factor: int) -> M:
        """Downsample the piano rolls by the given factor.

        Parameters
        ----------
        factor : int
            Ratio of the original resolution to the desired resolution.

        Returns
        -------
        Object itself.

        Note
        ----
        Attribute `resolution` will also be updated accordingly.

        """
        if self.resolution % factor > 0:
            raise ValueError(
                "Downsample factor must be a factor of the resolution."
            )
        self.resolution = self.resolution // factor
        for track in self.tracks:
            track.pianoroll = track.pianoroll[::factor]
        return self

    def pad(self: M, pad_length) -> M:
        """Pad the piano rolls.

        Notes
        -----
        The lengths of the resulting piano rolls are not guaranteed to
        be the same.

        Parameters
        ----------
        pad_length : int
            Length to pad along the time axis.

        Returns
        -------
        Object itself.

        See Also
        --------
        :meth:`pypianoroll.Multitrack.pad_to_multiple` : Pad the piano
          rolls so that their lengths are some multiples.
        :meth:`pypianoroll.Multitrack.pad_to_same` : Pad the piano rolls
          so that they have the same length.

        """
        for track in self.tracks:
            track.pad(pad_length)
        return self

    def pad_to_multiple(self: M, factor: int) -> M:
        """Pad the piano rolls so that their lengths are some multiples.

        Pad the piano rolls at the end along the time axis
        of the minimum length that makes the lengths of the resulting
        piano rolls multiples of `factor`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting piano rolls will
            be a multiple of.

        Returns
        -------
        Object itself.

        See Also
        --------
        :meth:`pypianoroll.Multitrack.pad` : Pad the piano rolls.
        :meth:`pypianoroll.Multitrack.pad_to_same` : Pad the piano rolls
          so that they have the same length.

        """
        for track in self.tracks:
            track.pad_to_multiple(factor)
        return self

    def pad_to_same(self: M) -> M:
        """Pad the piano rolls so that they have the same length.

        Pad shorter piano rolls at the end along the time
        axis so that the resulting piano rolls have the same length.

        Returns
        -------
        Object itself.

        See Also
        --------
        :meth:`pypianoroll.Multitrack.pad` : Pad the piano rolls.
        :meth:`pypianoroll.Multitrack.pad_to_multiple` : Pad the piano
          rolls so that their lengths are some multiples.

        """
        max_length = self.get_max_length()
        for track in self.tracks:
            if track.pianoroll.shape[0] < max_length:
                track.pad(max_length - track.pianoroll.shape[0])
        return self

    def transpose(self: M, semitone: int) -> M:
        """Transpose the piano rolls by a number of semitones.

        Parameters
        ----------
        semitone : int
            Number of semitones to transpose. A positive value raises
            the pitches, while a negative value lowers the pitches.

        Returns
        -------
        Object itself.

        Notes
        -----
        Drum tracks are skipped.

        """
        for track in self.tracks:
            if not track.is_drum:
                track.transpose(semitone)
        return self

    def trim_trailing_silence(self: M) -> M:
        """Trim the trailing silences of the piano rolls.

        All the piano rolls will have the same length after the
        trimming.

        Returns
        -------
        Object itself.

        """
        active_length = self.get_active_length()
        for track in self.tracks:
            track.pianoroll = track.pianoroll[:active_length]
        return self

    def save(self, path: str, compressed: bool = True):
        """Save to a NPZ file.

        Refer to :func:`pypianoroll.save` for full documentation.

        """
        save(path, self, compressed=compressed)

    def write(self, path: str):
        """Write to a MIDI file.

        Refer to :func:`pypianoroll.write` for full documentation.

        """
        return write(path, self)

    def to_pretty_midi(self, **kwargs):
        """Return as a PrettyMIDI object.

        Refer to :func:`pypianoroll.to_pretty_midi` for full
        documentation.

        """
        return to_pretty_midi(self, **kwargs)

    def plot(self, **kwargs):
        """Plot the multitrack and/or save a plot of it.

        Refer to :func:`pypianoroll.plot_multitrack` for full
        documentation.

        """
        return plot_multitrack(self, **kwargs)

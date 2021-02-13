"""Class for multitrack piano rolls.

Class
-----

- Multitrack

Variable
--------

- DEFAULT_RESOLUTION

"""
from typing import Optional, Sequence, TypeVar

import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray

from .outputs import save, to_pretty_midi, write
from .track import BinaryTrack, StandardTrack, Track
from .visualization import plot_multitrack

__all__ = [
    "Multitrack",
    "DEFAULT_RESOLUTION",
]

DEFAULT_RESOLUTION = 24

_Multitrack = TypeVar("_Multitrack", bound="Multitrack")


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
    tracks : sequence of :class:`pypianoroll.Track`, optional
        Music tracks.

    """

    def __init__(
        self,
        name: Optional[str] = None,
        resolution: Optional[int] = None,
        tempo: Optional[ndarray] = None,
        downbeat: Optional[ndarray] = None,
        tracks: Optional[Sequence[Track]] = None,
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
        elif downbeat.dtype == np.bool_:
            self.downbeat = downbeat
        else:
            self.downbeat = np.asarray(downbeat).astype(bool)

        if tracks is None:
            self.tracks = []
        elif isinstance(tracks, list):
            self.tracks = tracks
        else:
            self.tracks = list(tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, val):
        return self.tracks[val]

    def __repr__(self) -> str:
        to_join = [
            f"name={repr(self.name)}",
            f"resolution={repr(self.resolution)}",
        ]
        if self.tempo is not None:
            to_join.append(
                f"tempo=array(shape={self.tempo.shape}, "
                f"dtype={self.tempo.dtype})"
            )
        if self.downbeat is not None:
            to_join.append(
                f"downbeat=array(shape={self.downbeat.shape}, "
                f"dtype={self.downbeat.dtype})"
            )
        to_join.append(f"tracks={repr(self.tracks)}")
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
        if getattr(self, attr) is None:
            if attr == "resolution":
                raise TypeError(f"`{attr}` must not be None.")
            return

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

    def validate(self: _Multitrack, attr=None) -> _Multitrack:
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

    def get_length(self) -> int:
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
            now_length = track.get_length()
            if active_length < track.get_length():
                active_length = now_length
        return active_length

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

    def set_nonzeros(self: _Multitrack, value: int) -> _Multitrack:
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
            if isinstance(track, (StandardTrack, BinaryTrack)):
                self.tracks[i] = track.set_nonzeros(value)
        return self

    def set_resolution(
        self: _Multitrack, resolution: int, rounding: Optional[str] = "round",
    ) -> _Multitrack:
        """Set the resolution.

        Parameters
        ----------
        resolution : int
            Target resolution.
        rounding : {'round', 'ceil', 'floor'}
            Rounding mode. Defaults to 'round'.

        Returns
        -------
        Object itself.

        """
        for track in self.tracks:
            time, pitch = track.pianoroll.nonzero()
            if len(time) < 1:
                continue
            if track.pianoroll.dtype == np.bool_:
                value = 1
            else:
                value = track.pianoroll[time, pitch]
            factor = resolution / self.resolution
            if rounding == "round":
                time = np.round(time * factor).astype(int)
            elif rounding == "ceil":
                time = np.ceil(time * factor).astype(int)
            elif rounding == "floor":
                time = np.floor(time * factor).astype(int)
            else:
                raise ValueError(
                    "`rounding` must be one of 'round', 'ceil' or 'floor', "
                    f"not {rounding}."
                )
            track.pianoroll = np.zeros(
                (time[-1] + 1, 128), track.pianoroll.dtype
            )
            track.pianoroll[time, pitch] = value
        self.resolution = resolution
        return self

    def count_downbeat(self) -> int:
        """Return the number of downbeats.

        Returns
        -------
        int
            Number of downbeats.

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
        max_length = self.get_max_length()
        pianorolls = []
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

    def blend(self, mode: Optional[str] = None) -> ndarray:
        """Return the blended pianoroll.

        Parameters
        ----------
        mode : {'sum', 'max', 'any'}, optional
            Blending strategy to apply along the track axis. For 'sum'
            mode, integer summation is performed for binary piano rolls.
            Defaults to 'sum'.

        Returns
        -------
        ndarray, shape=(?, 128)
            Blended piano roll.

        """
        stacked = self.stack()
        if mode is None or mode.lower() == "sum":
            return np.sum(stacked, axis=0).clip(0, 127).astype(np.uint8)
        if mode.lower() == "any":
            return np.any(stacked, axis=0)
        if mode.lower() == "max":
            return np.max(stacked, axis=0)
        raise ValueError("`mode` must be one of 'max', 'sum' or 'any'.")

    def copy(self):
        """Return a copy of the multitrack.

        Returns
        -------
        A copy of the object itself.

        Notes
        -----
        Arrays are copied using :func:`numpy.copy`.

        """
        return Multitrack(
            name=self.name,
            resolution=self.resolution,
            tempo=None if self.tempo is None else self.tempo.copy(),
            downbeat=None if self.downbeat is None else self.downbeat.copy(),
            tracks=[track.copy() for track in self.tracks],
        )

    def append(self: _Multitrack, track: Track) -> _Multitrack:
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

    def binarize(self: _Multitrack, threshold: float = 0) -> _Multitrack:
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

    def clip(
        self: _Multitrack, lower: int = 0, upper: int = 127
    ) -> _Multitrack:
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

        Note
        ----
        Only affect StandardTrack instances.

        """
        for track in self.tracks:
            if isinstance(track, StandardTrack):
                track.clip(lower, upper)
        return self

    def pad(self: _Multitrack, pad_length) -> _Multitrack:
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

    def pad_to_multiple(self: _Multitrack, factor: int) -> _Multitrack:
        """Pad the piano rolls so that their lengths are some multiples.

        Pad the piano rolls at the end along the time axis of the
        minimum length that makes the lengths of the resulting piano
        rolls multiples of `factor`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting piano rolls will
            be a multiple of.

        Returns
        -------
        Object itself.

        Notes
        -----
        Lengths of the resulting piano rolls are necessarily the same.

        See Also
        --------
        :meth:`pypianoroll.Multitrack.pad` : Pad the piano rolls.
        :meth:`pypianoroll.Multitrack.pad_to_same` : Pad the piano rolls
          so that they have the same length.

        """
        for track in self.tracks:
            track.pad_to_multiple(factor)
        return self

    def pad_to_same(self: _Multitrack) -> _Multitrack:
        """Pad the piano rolls so that they have the same length.

        Pad shorter piano rolls at the end along the time axis so that
        the resulting piano rolls have the same length.

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

    def remove_empty(self: _Multitrack) -> _Multitrack:
        """Remove tracks with empty pianorolls."""
        self.tracks = [
            track for track in self.tracks if not np.any(track.pianoroll)
        ]
        return self

    def transpose(self: _Multitrack, semitone: int) -> _Multitrack:
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

    def trim(
        self: _Multitrack,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> _Multitrack:
        """Trim the trailing silences of the piano rolls.

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
        if end is None:
            end = self.get_length()
        if self.tempo is not None:
            self.tempo = self.tempo[start:end]
        if self.downbeat is not None:
            self.downbeat = self.downbeat[start:end]
        for track in self.tracks:
            track.trim(start=start, end=end)
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

    def plot(self, axs: Optional[Sequence[Axes]] = None, **kwargs) -> ndarray:
        """Plot the multitrack piano roll.

        Refer to :func:`pypianoroll.plot_multitrack` for full
        documentation.

        """
        return plot_multitrack(self, axs, **kwargs)

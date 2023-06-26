"""Class for multitrack piano rolls.

Class
-----

- Multitrack

Variable
--------

- DEFAULT_RESOLUTION

"""
from typing import List, Sequence, TypeVar, Union

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

MultitrackType = TypeVar("MultitrackType", bound="Multitrack")


def _round_time(time, factor, rounding):
    if rounding == "round":
        return np.round(time * factor).astype(int)
    if rounding == "ceil":
        return np.ceil(time * factor).astype(int)
    if rounding == "floor":
        return np.floor(time * factor).astype(int)
    raise ValueError(
        "`rounding` must be one of 'round', 'ceil' or 'floor', "
        f"not {rounding}."
    )


class Multitrack:
    """A container for multitrack piano rolls.

    This is the core class of Pypianoroll. A Multitrack object can be
    constructed in the following ways.

    - :meth:`pypianoroll.Multitrack`: Construct by setting values for
      attributes
    - :meth:`pypianoroll.read`: Read from a MIDI file
    - :meth:`pypianoroll.from_pretty_midi`: Convert from a
      :class:`pretty_midi.PrettyMIDI` object
    - :func:`pypianoroll.load`: Load from a JSON or a YAML file saved by
      :func:`pypianoroll.save`

    Attributes
    ----------
    name : str, optional
        Multitrack name.
    resolution : int, default: `pypianoroll.DEFAULT_RESOLUTION` (24)
        Time steps per quarter note.
    tempo : ndarray, dtype=float, shape=(?, 1), optional
        Tempo (in qpm) at each time step. Length is the total number
        of time steps. Cast to float if not of float type. Alternatively,
        enter a single float or integer and the array will be generated.
    beat : ndarray, dtype=bool, shape=(?, 1), optional
        A boolean array that indicates whether the time step contains a
        beat. Length is the total number of time steps. Cast to bool if
        not of bool type.
    downbeat : ndarray, dtype=bool, shape=(?, 1), optional
        A boolean array that indicates whether the time step contains a
        downbeat, i.e., the first time step of a measure. Length is the
        total number of time steps. Cast to bool if not of bool type.
    tracks : sequence of :class:`pypianoroll.Track`, default: []
        Music tracks.

    """

    def __init__(
        self,
        name: str = None,
        resolution: int = None,
        tempo: Union[ndarray, int, float] = None,
        beat: ndarray = None,
        downbeat: ndarray = None,
        tracks: Sequence[Track] = None,
    ):
        self.name = name

        if resolution is not None:
            self.resolution = resolution
        else:
            self.resolution = DEFAULT_RESOLUTION

        if tracks is None:
            self.tracks = []
        elif isinstance(tracks, list):
            self.tracks = tracks
        else:
            self.tracks = list(tracks)

        if tempo is None:
            self.tempo = None
        elif isinstance(tempo, int) or isinstance(tempo, float):
            self.tempo = np.tile(tempo, (self.get_max_length(), 1)).astype(float)
        elif np.issubdtype(tempo.dtype, np.floating):
            self.tempo = tempo
        else:
            self.tempo = np.asarray(tempo).astype(float)

        if beat is None:
            self.beat = None
        elif beat.dtype == np.bool_:
            self.beat = beat
        else:
            self.beat = np.asarray(beat).astype(bool)

        if downbeat is None:
            self.downbeat = None
        elif downbeat.dtype == np.bool_:
            self.downbeat = downbeat
        else:
            self.downbeat = np.asarray(downbeat).astype(bool)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, key: int) -> Track:
        return self.tracks[key]

    def __setitem__(self, key: int, value: Track):
        self.tracks[key] = value

    def __delitem__(self, key: int):
        del self.tracks[key]

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
        if self.beat is not None:
            to_join.append(
                f"beat=array(shape={self.beat.shape}, dtype={self.beat.dtype})"
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
        elif attr == "beat":
            if not isinstance(self.beat, np.ndarray):
                raise TypeError("`beat` must be a NumPy array.")
            if not np.issubdtype(self.beat.dtype, bool):
                raise TypeError(
                    "`beat` must be of data type bool, but got data type"
                    f"{self.beat.dtype}."
                )
        elif attr == "downbeat":
            if not isinstance(self.downbeat, np.ndarray):
                raise TypeError("`downbeat` must be a NumPy array.")
            if not np.issubdtype(self.downbeat.dtype, bool):
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
            attributes = (
                "name",
                "resolution",
                "tempo",
                "beat",
                "downbeat",
                "tracks",
            )
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
            if self.tempo.ndim != 2:
                raise ValueError("`tempo` must be a 2D NumPy array of shape (?,1)")
            if np.any(self.tempo <= 0.0):
                raise ValueError("`tempo` must contain only positive numbers.")
        elif attr == "beat":
            if self.beat.ndim != 1:
                raise ValueError("`beat` must be a 1D NumPy array.")
        elif attr == "downbeat":
            if self.downbeat.ndim != 2 or self.downbeat.shape[1] != 1:
                raise ValueError("`downbeat` must be a 2D NumPy array of shape (?,1).")
        elif attr == "tracks":
            for track in self.tracks:
                track.validate()

    def validate(self: MultitrackType, attr=None) -> MultitrackType:
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
            attributes = (
                "name",
                "resolution",
                "tempo",
                "beat",
                "downbeat",
                "tracks",
            )
            for attribute in attributes:
                self._validate(attribute)
        else:
            self._validate(attr)
        return self

    def is_valid_type(self, attr: str = None) -> bool:
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

    def is_valid(self, attr: str = None) -> bool:
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

    def get_end_time(self) -> int:
        """Return the end time of the multitrack.

        Returns
        -------
        int
            Maximum length (in time steps) of the tempo, beat, downbeat
            arrays and all piano rolls.

        """
        end_time = self.get_max_length()
        if self.tempo is not None and end_time < self.tempo.shape[0]:
            end_time = self.tempo.shape[0]
        if self.beat is not None and end_time < self.beat.shape[0]:
            end_time = self.beat.shape[0]
        if self.downbeat is not None and end_time < self.downbeat.shape[0]:
            end_time = self.downbeat.shape[0]
        return end_time

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

    def get_beat_steps(self) -> ndarray:
        """Return the indices of time steps that contain beats.

        Returns
        -------
        ndarray, dtype=int
            Indices of time steps that contain beats.

        """
        if self.beat is None:
            return np.array([])
        return np.nonzero(self.beat)[0]

    def get_downbeat_steps(self) -> ndarray:
        """Return the indices of time steps that contain downbeats.

        Returns
        -------
        ndarray, dtype=int
            Indices of time steps that contain downbeats.

        """
        if self.downbeat is None:
            return np.array([])
        return np.nonzero(self.downbeat)[0]

    def set_nonzeros(self: MultitrackType, value: int) -> MultitrackType:
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
        self: MultitrackType, resolution: int, rounding: str = "round"
    ) -> MultitrackType:
        """Set the resolution.

        Parameters
        ----------
        resolution : int
            Target resolution.
        rounding : {'round', 'ceil', 'floor'}, default: 'round'
            Rounding mode.

        Returns
        -------
        Object itself.

        """
        factor = resolution / self.resolution
        # Get the end time
        end_time = self.get_end_time()
        rounded_end_time = _round_time(end_time, factor, rounding)
        # Beat array
        beats = self.get_beat_steps()
        beats = _round_time(beats, factor, rounding)
        self.beat = np.zeros((rounded_end_time + 1, 1), bool)
        self.beat[beats] = True
        # Downbeat array
        downbeats = self.get_downbeat_steps()
        downbeats = _round_time(downbeats, factor, rounding)
        self.downbeat = np.zeros((rounded_end_time + 1, 1), bool)
        self.downbeat[downbeats] = True
        # Iterate over each track
        for track in self.tracks:
            time, pitch = track.pianoroll.nonzero()
            if len(time) < 1:
                continue
            if track.pianoroll.dtype == np.bool_:
                value = 1
            else:
                value = track.pianoroll[time, pitch]
            rounded_time = _round_time(time, factor, rounding)
            track.pianoroll = np.zeros(
                (rounded_end_time + 1, 128), track.pianoroll.dtype
            )
            track.pianoroll[rounded_time, pitch] = value
        # Set the new resolution
        self.resolution = resolution
        return self

    def count_beat(self) -> int:
        """Return the number of beats.

        Returns
        -------
        int
            Number of beats.

        Note
        ----
        Return value is calculated based only on the attribute `beat`.

        """
        if self.beat is None:
            return 0
        return np.count_nonzero(self.beat)

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
        if self.downbeat is None:
            return 0
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
                    track.pianoroll,
                    ((0, pad_length), (0, 0)),
                    "constant",
                )
                pianorolls.append(padded)
            else:
                pianorolls.append(track.pianoroll)
        return np.stack(pianorolls)

    def blend(self, mode: str = None) -> ndarray:
        """Return the blended pianoroll.

        Parameters
        ----------
        mode : {'sum', 'max', 'any'}, default: 'sum'
            Blending strategy to apply along the track axis. For 'sum'
            mode, integer summation is performed for binary piano rolls.

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
            beat=None if self.beat is None else self.beat.copy(),
            downbeat=None if self.downbeat is None else self.downbeat.copy(),
            tracks=[track.copy() for track in self.tracks],
        )

    def append(self: MultitrackType, track: Track) -> MultitrackType:
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

    def binarize(self: MultitrackType, threshold: float = 0) -> MultitrackType:
        """Binarize the piano rolls.

        Parameters
        ----------
        threshold : int or float, default: 0
            Threshold to binarize the piano rolls.

        Returns
        -------
        Object itself.

        """
        for i, track in enumerate(self.tracks):
            if isinstance(track, StandardTrack):
                self.tracks[i] = track.binarize(threshold)
        return self

    def clip(
        self: MultitrackType, lower: int = 0, upper: int = 127
    ) -> MultitrackType:
        """Clip (limit) the the piano roll into [`lower`, `upper`].

        Parameters
        ----------
        lower : int, default: 0
            Lower bound.
        upper : int, default: 127
            Upper bound.

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

    def pad(self: MultitrackType, pad_length) -> MultitrackType:
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

    def pad_to_multiple(self: MultitrackType, factor: int) -> MultitrackType:
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

    def pad_to_same(self: MultitrackType) -> MultitrackType:
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

    def remove_empty(self: MultitrackType) -> MultitrackType:
        """Remove tracks with empty pianorolls."""
        self.tracks = [
            track for track in self.tracks if not np.any(track.pianoroll)
        ]
        return self

    def transpose(self: MultitrackType, semitone: int) -> MultitrackType:
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
        self: MultitrackType, start: int = None, end: int = None
    ) -> MultitrackType:
        """Trim the trailing silences of the piano rolls.

        Parameters
        ----------
        start : int, default: 0
            Start time.
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
        if self.beat is not None:
            self.beat = self.beat[start:end]
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

    def plot(self, axs: Sequence[Axes] = None, **kwargs) -> List[Axes]:
        """Plot the multitrack piano roll.

        Refer to :func:`pypianoroll.plot_multitrack` for full
        documentation.

        """
        return plot_multitrack(self, axs, **kwargs)

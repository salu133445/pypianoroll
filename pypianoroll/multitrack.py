"""Multitrack class.

This module defines the core class of Pypianoroll---the Multitrack
class, a container for multitrack piano rolls.

"""
import json
import zipfile
from copy import deepcopy

import numpy as np
import pretty_midi
from scipy.sparse import csc_matrix

from .track import Track
from .visualization import plot_multitrack

__all__ = ["Multitrack"]

DEFAULT_RESOLUTION = 24
DEFAULT_TEMPO = 120


def decompose_sparse(matrix, name):
    """Decompose a matrix to sparse components and return as a dictionary.

    Convert a matrix to a :class:`scipy.sparse.csc_matrix` object. Return
    its component arrays as a dictionary with key as `name` suffixed with
    their component types.

    """
    csc = csc_matrix(matrix)
    return {
        name + "_csc_data": csc.data,
        name + "_csc_indices": csc.indices,
        name + "_csc_indptr": csc.indptr,
        name + "_csc_shape": csc.shape,
    }


def reconstruct_sparse(data_dict, name):
    """Reconstruct a matrix from a dictionary return by `_decompose_sparse`."""
    sparse_matrix = csc_matrix(
        (
            data_dict[name + "_csc_data"],
            data_dict[name + "_csc_indices"],
            data_dict[name + "_csc_indptr"],
        ),
        shape=data_dict[name + "_csc_shape"],
    )
    return sparse_matrix.toarray()


class Multitrack:
    """A container for multitrack piano rolls.

    This is the core class of Pypianoroll.

    Attributes
    ----------
    resolution : int
        Time steps per quarter note.
    tempo : ndarray, dtype={int, float}, shape=(?, 1), optional
        Tempo (in qpm) at each time step. The length is the total number of
        time steps.
    downbeat : ndarray, dtype=bool, shape=(?, 1), optional
        A boolean array that indicates whether the time step contains a
        downbeat (i.e., the first time step of a bar). The length is the total
        number of time steps.
    name : str, optional
        Multitrack name.
    tracks : list of :class:`pypianoroll.Track` objects, optional
        Music tracks.

    """

    def __init__(
        self,
        resolution=None,
        tempo=None,
        downbeat=None,
        name=None,
        tracks=None,
    ):
        if resolution is None:
            self.resolution = DEFAULT_RESOLUTION
        else:
            self.resolution = resolution

        self.tempo = np.asarray(tempo) if tempo is not None else None

        if downbeat is None:
            self.downbeat = None
        else:
            downbeat = np.asarray(downbeat)
            if np.issubdtype(downbeat.dtype, np.integer):
                self.downbeat = np.zeros((max(downbeat) + 1, 1), bool)
                self.downbeat[downbeat] = True
            else:
                self.downbeat = downbeat

        self.name = name
        self.tracks = tracks if tracks is not None else []

    def __len__(self):
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
                tracks=tracks,
                tempo=tempo,
                downbeat=downbeat,
                resolution=self.resolution,
                name=self.name,
            )

        if isinstance(val, list):
            tracks = [self.tracks[i] for i in val]
        else:
            tracks = self.tracks[val]

        return Multitrack(
            tracks=tracks,
            tempo=self.tempo,
            downbeat=self.downbeat,
            resolution=self.resolution,
            name=self.name,
        )

    def __repr__(self):
        to_join = []
        if self.name is not None:
            to_join.append("name=" + repr(self.name))
        to_join.append("resolution=" + repr(self.resolution))
        if self.tempo.size:
            to_join.append("tempo=[" + repr(self.tempo[0]) + ", ...]")
        if self.downbeat.size:
            to_join.append("downbeat=[" + repr(self.downbeat[0]) + ", ...]")
        if self.tracks:
            to_join.append("tracks=" + repr(self.tracks))
        return "Multitrack(" + ", ".join(to_join) + ")"

    def validate(self):
        """Raise a proper error if any attribute is invalid."""
        # Resolution
        if not isinstance(self.resolution, int):
            raise TypeError("`resolution` must be of type int.")
        if self.resolution < 1:
            raise ValueError("`resolution` must be a positive integer.")

        # Tempo
        if not isinstance(self.tempo, np.ndarray):
            raise TypeError("`tempo` must be a NumPy array.")
        if not np.issubdtype(self.tempo.dtype, np.number):
            raise TypeError(
                "Data type of `tempo` must be a subdtype of np.number."
            )
        if self.tempo.ndim != 1:
            raise ValueError("`tempo` must be a 1D NumPy array.")
        if np.any(self.tempo <= 0.0):
            raise ValueError("`tempo` should contain only positive numbers.")

        # Downbeat
        if self.downbeat is not None:
            if not isinstance(self.downbeat, np.ndarray):
                raise TypeError("`downbeat` must be a NumPy array.")
            if not np.issubdtype(self.downbeat.dtype, np.bool_):
                raise TypeError("Data type of `downbeat` must be bool.")
            if self.downbeat.ndim != 1:
                raise ValueError("`downbeat` must be a 1D NumPy array.")

        # Name
        if not isinstance(self.name, str):
            raise TypeError("`name` must be of type str.")

        # Tracks
        for track in self.tracks:
            if not isinstance(track, Track):
                raise TypeError(
                    "`tracks` must be a list of `pypianoroll.Track` instances."
                )
            track.validate()

    def is_binarized(self):
        """Return True if all piano rolls are binarized, otherwise False."""
        for track in self.tracks:
            if not track.is_binarized():
                return False
        return True

    def get_active_length(self):
        """Return the maximum active length of the piano rolls (in time steps).

        The active length is defined as the length of the piano roll without
        trailing silence.

        Returns
        -------
        int
            The maximum active length of the piano rolls (in time steps).

        """
        active_length = 0
        for track in self.tracks:
            now_length = track.get_active_length()
            if active_length < track.get_active_length():
                active_length = now_length
        return active_length

    def get_active_pitch_range(self):
        """Return the active pitch range as a tuple (lowest, highest).

        Returns
        -------
        lowest : int
            Lowest active pitch in all the piano rolls.
        highest : int
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

    def get_downbeat_steps(self):
        """Return the indices of time steps that contain downbeats.

        Returns
        -------
        downbeat_steps : list
            Indices of time steps that contain downbeats.

        """
        if self.downbeat is None:
            return []
        downbeat_steps = np.nonzero(self.downbeat)[0].tolist()
        return downbeat_steps

    def get_empty_tracks(self):
        """Return the indices of tracks with empty pianorolls.

        Returns
        -------
        list
            Indices of tracks with empty pianorolls.

        """
        indices = []
        for i, track in enumerate(self.tracks):
            if not np.any(track.pianoroll):
                indices.append(i)
        return indices

    def get_max_length(self):
        """Return the maximum length of the piano rolls (in time steps).

        Returns
        -------
        max_length : int
            Maximum length of the piano rolls (in time step).

        """
        max_length = 0
        for track in self.tracks:
            if max_length < track.pianoroll.shape[0]:
                max_length = track.pianoroll.shape[0]
        return max_length

    def get_merged_pianoroll(self, mode="sum"):
        """Return the merged piano roll.

        Parameters
        ----------
        mode : {'sum', 'max', 'any'}
            A string that indicates the merging strategy to apply along the
            track axis. Default to 'sum'.

            - In 'sum' mode, the merged pianoroll is the sum of all the
              pianorolls. Note that for binarized pianorolls, integer summation
              is performed.
            - In 'max' mode, for each pixel, the maximum value among all the
              pianorolls is assigned to the merged pianoroll.
            - In 'any' mode, the value of a pixel in the merged pianoroll is
              True if any of the pianorolls has nonzero value at that pixel;
              False if all pianorolls are inactive (zero-valued) at that pixel.

        Returns
        -------
        ndarray, shape=(?, 128)
            Merged piano roll.

        """
        stacked = self.get_stacked_pianoroll()

        if mode == "any":
            merged = np.any(stacked, axis=2)
        elif mode == "sum":
            merged = np.sum(stacked, axis=2)
        elif mode == "max":
            merged = np.max(stacked, axis=2)
        else:
            raise ValueError("`mode` must be one of {'max', 'sum', 'any'}.")

        return merged

    def get_stacked_pianoroll(self):
        """Return a stacked multitrack piano-roll as a tensor.

        The shape of the return array is (n_time_steps, 128, n_tracks).

        Returns
        -------
        ndarray, shape=(?, 128, ?)
            Stacked piano roll.

        """
        multitrack = deepcopy(self)
        multitrack.pad_to_same()
        stacked = np.stack(
            [track.pianoroll for track in multitrack.tracks], -1
        )
        return stacked

    def append(self, track):
        """Append a :class:`multitrack.Track` object to the track list.

        Parameters
        ----------
        track : pianoroll.Track
            Track to append to the track list.

        """
        self.tracks.append(track)
        return self

    def assign_constant(self, value):
        """Assign a constant value to all nonzeros entries of the piano rolls.

        If a piano roll is not binarized, its data type will be preserved. If a
        piano roll is binarized, cast it to the dtype of `value`.

        Arguments
        ---------
        value : int or float
            Value to assign to all the nonzero entries in the piano rolls.

        """
        for track in self.tracks:
            track.assign_constant(value)

    def binarize(self, threshold=0):
        """Binarize the piano rolls.

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano rolls. Defaults to zero.

        """
        for track in self.tracks:
            track.binarize(threshold)
        return self

    def clip(self, lower=0, upper=127):
        """Clip the piano rolls by a lower bound and an upper bound.

        Parameters
        ----------
        lower : int or float
            Lower bound to clip the piano rolls. Defaults to 0.
        upper : int or float
            Upper bound to clip the piano rolls. Defaults to 127.

        """
        for track in self.tracks:
            track.clip(lower, upper)
        return self

    def downsample(self, factor):
        """Downsample the piano rolls by the given factor.

        Attribute `resolution` will be updated accordingly as well.

        Parameters
        ----------
        factor : int
            Ratio of the original resolution to the desired resolution.

        """
        if self.resolution % factor > 0:
            raise ValueError(
                "Downsample factor must be a factor of the resolution."
            )
        self.resolution = self.resolution // factor
        for track in self.tracks:
            track.pianoroll = track.pianoroll[::factor]
        return self

    def count_downbeat(self):
        """Return the number of down beats.

        The return value is calculated based solely on attribute `downbeat`.

        Returns
        -------
        int
            Number of down beats.

        """
        return np.count_nonzero(self.downbeat)

    def merge_tracks(
        self,
        track_indices=None,
        mode="sum",
        program=0,
        is_drum=False,
        name="merged",
        remove_source=False,
    ):
        """Merge certain tracks into a single track.

        Merge the piano rolls of certain tracks (specified by `track_indices`).
        The merged track will be appended to the end of the track list.

        Parameters
        ----------
        track_indices : list
            Indices of tracks to be merged. Defaults to merge all the tracks.
        mode : {'sum', 'max', 'any'}
            A string that indicates the merging strategy to apply along the
            track axis. Default to 'sum'.

            - In 'sum' mode, the merged pianoroll is the sum of the collected
              pianorolls. Note that for binarized pianorolls, integer summation
              is performed.
            - In 'max' mode, for each pixel, the maximum value among the
              collected pianorolls is assigned to the merged pianoroll.
            - In 'any' mode, the value of a pixel in the merged pianoroll is
              True if any of the collected pianorolls has nonzero value at that
              pixel; False if all the collected pianorolls are inactive
              (zero-valued) at that pixel.

        program : int, 0-127, optional
            Program number according to General MIDI specification [1].
            Defaults to 0 (Acoustic Grand Piano).
        is_drum : bool, optional
            Whether it is a percussion track. Defaults to False.
        name : str, optional
            Track name. Defaults to `merged`.
        remove_source : bool
            Whether to remove the source tracks from the track list. Defaults
            to False.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set

        """
        if mode not in ("max", "sum", "any"):
            raise ValueError("`mode` must be one of {'max', 'sum', 'any'}.")

        merged = self[track_indices].get_merged_pianoroll(mode)

        merged_track = Track(merged, program, is_drum, name)
        self.append(merged_track)

        if remove_source:
            self.remove_tracks(track_indices)

        return self

    def pad(self, pad_length):
        """Pad the piano rolls with zeros at the end along the time axis.

        Notes
        -----
        The lengths of the resulting piano rolls are not guaranteed to be
        the same. See :meth:`pypianoroll.Multitrack.pad_to_same`.

        Parameters
        ----------
        pad_length : int
            Length to pad along the time axis with zeros.

        """
        for track in self.tracks:
            track.pad(pad_length)
        return self

    def pad_to_multiple(self, factor):
        """Pad the piano rolls along the time axis to a multiple of `factor`.

        Pad the piano rolls with zeros at the end along the time axis of the
        minimum length that makes the lengths of the resulting piano rolls
        multiples of `factor`.

        Notes
        -----
        The resulting pianoroll lengths are not guaranteed to be the same. See
        :meth:`pypianoroll.Multitrack.pad_to_same`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting pianorolls will be a
            multiple of.

        """
        for track in self.tracks:
            track.pad_to_multiple(factor)
        return self

    def pad_to_same(self):
        """Pad piano rolls along the time axis to have the same length.

        Pad shorter piano rolls with zeros at the end along the time axis so
        that the resulting piano rolls have the same length.

        """
        max_length = self.get_max_length()
        for track in self.tracks:
            if track.pianoroll.shape[0] < max_length:
                track.pad(max_length - track.pianoroll.shape[0])
        return self

    def remove_empty_tracks(self):
        """Remove tracks with empty pianorolls."""
        self.remove_tracks(self.get_empty_tracks())

    def remove_tracks(self, track_indices):
        """Remove certain tracks.

        Parameters
        ----------
        track_indices : list
            Indices of the tracks to remove.

        """
        if isinstance(track_indices, int):
            track_indices = [track_indices]
        self.tracks = [
            track
            for idx, track in enumerate(self.tracks)
            if idx not in track_indices
        ]
        return self

    def transpose(self, semitone):
        """Transpose the piano rolls by a number of semitones.

        Positive values are for a higher key, while negative values are for
        a lower key. Drum tracks are ignored.

        Parameters
        ----------
        semitone : int
            Number of semitones to transpose the piano rolls.

        """
        for track in self.tracks:
            if not track.is_drum:
                track.transpose(semitone)

    def trim_trailing_silence(self):
        """Trim the trailing silences of the piano rolls.

        All the piano rolls will have the same length after the trimming.

        """
        active_length = self.get_active_length()
        for track in self.tracks:
            track.pianoroll = track.pianoroll[:active_length]
        return self

    def save(self, path, compressed=True):
        """Save to a (compressed) NPZ file.

        This could be later loaded by :func:`pypianoroll.load`.

        Parameters
        ----------
        path : str
            Path to the NPZ file to save.
        compressed : bool
            Whether to save to a compressed NPZ file. Defaults to True.

        Notes
        -----
        To reduce the file size, the piano rolls are first converted to
        instances of :class:`scipy.sparse.csc_matrix`. The component arrays
        are then collected and saved to a npz file.

        """
        info_dict = {
            "resolution": self.resolution,
            "name": self.name,
        }

        array_dict = {}
        if self.tempo is not None:
            array_dict["tempo"] = self.tempo
        if self.downbeat is not None:
            array_dict["downbeat"] = self.downbeat

        for idx, track in enumerate(self.tracks):
            array_dict.update(
                decompose_sparse(track.pianoroll, "pianoroll_" + str(idx))
            )
            info_dict[str(idx)] = {
                "program": track.program,
                "is_drum": track.is_drum,
                "name": track.name,
            }

        if not path.endswith(".npz"):
            path += ".npz"
        if compressed:
            np.savez_compressed(path, **array_dict)
        else:
            np.savez(path, **array_dict)

        compression = (
            zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        )
        with zipfile.ZipFile(path, "a") as zip_file:
            zip_file.writestr("info.json", json.dumps(info_dict), compression)

    def to_pretty_midi(self, default_tempo=None, default_velocity=64):
        """Convert to a :class:`pretty_midi.PrettyMIDI` object.

        Notes
        -----
        - Tempo changes are not supported by now.
        - The velocities of the converted pianorolls are clipped to [0, 127].
        - Adjacent nonzero values of the same pitch will be considered a single
          note with their mean as its velocity.

        Parameters
        ----------
        default_tempo : int
            Default tempo to use. Defaults to the first element of attribute
            `tempo`.
        default_velocity : int
            Default velocity to assign to binarized tracks. Defaults to 64.

        Returns
        -------
        pm : `pretty_midi.PrettyMIDI` object
            Converted :class:`pretty_midi.PrettyMIDI` instance.

        """
        # TODO: Add downbeat support -> time signature change events
        # TODO: Add tempo support -> tempo change events
        if default_tempo is not None:
            tempo = default_tempo
        elif self.tempo:
            tempo = self.tempo[0]
        else:
            tempo = DEFAULT_TEMPO

        # Create a PrettyMIDI instance
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Compute length of a time step
        time_step_length = 60.0 / tempo / self.resolution

        for track in self.tracks:
            instrument = pretty_midi.Instrument(
                program=track.program, is_drum=track.is_drum, name=track.name
            )
            copied = track.copy()
            if copied.is_binarized():
                copied.assign_constant(default_velocity)
            copied.clip()
            clipped = copied.pianoroll.astype(np.uint8)
            binarized = clipped > 0
            padded = np.pad(binarized, ((1, 1), (0, 0)), "constant")
            diff = np.diff(padded.astype(np.int8), axis=0)

            positives = np.nonzero((diff > 0).T)
            pitches = positives[0]
            note_ons = positives[1]
            note_on_times = time_step_length * note_ons
            note_offs = np.nonzero((diff < 0).T)[1]
            note_off_times = time_step_length * note_offs

            for idx, pitch in enumerate(pitches):
                velocity = np.mean(
                    clipped[note_ons[idx] : note_offs[idx], pitch]
                )
                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=pitch,
                    start=note_on_times[idx],
                    end=note_off_times[idx],
                )
                instrument.notes.append(note)

            instrument.notes.sort(key=lambda x: x.start)
            pm.instruments.append(instrument)

        return pm

    def write(self, path):
        """Write to a MIDI file.

        Parameters
        ----------
        path : str
            Path to the MIDI file to write.

        """
        if not path.lower().endswith((".mid", ".midi")):
            path = path + ".mid"
        pm = self.to_pretty_midi()
        pm.write(path)

    def plot(self, **kwargs):
        """Plot the multitrack and/or save a plot of it.

        See :func:`pypianoroll.plot_multitrack` for full documentation.

        """
        return plot_multitrack(self, **kwargs)

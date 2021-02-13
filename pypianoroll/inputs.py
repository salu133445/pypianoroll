"""Input interfaces.

Functions
---------

- load
- from_pretty_midi
- read

"""
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pretty_midi import PrettyMIDI

from .multitrack import DEFAULT_RESOLUTION, Multitrack
from .track import BinaryTrack, StandardTrack, Track
from .utils import reconstruct_sparse

__all__ = ["load", "from_pretty_midi", "read"]


def load(path: Union[str, Path]) -> Multitrack:
    """Load a NPZ file into a Multitrack object.

    Supports only files previously saved by :func:`pypianoroll.save`.

    Parameters
    ----------
    path : str or Path
        Path to the file to load.

    See Also
    --------
    :func:`pypianoroll.save` : Save a Multitrack object to a NPZ file.
    :func:`pypianoroll.read` : Read a MIDI file into a Multitrack
      object.

    """
    with np.load(path) as loaded:
        if "info.json" not in loaded:
            raise RuntimeError("Cannot find `info.json` in the NPZ file.")

        # Load the info dictionary
        info_dict = json.loads(loaded["info.json"].decode("utf-8"))

        # Get the resolution
        resolution = info_dict.get("resolution")

        # Look for `beat_resolution` for backward compatibility
        if resolution is None:
            resolution = info_dict.get("beat_resolution")
            if resolution is None:
                raise RuntimeError(
                    "Cannot find `resolution` or `beat_resolution` in "
                    "`info.json`."
                )

        # Load the tracks
        idx = 0
        tracks = []
        while str(idx) in info_dict:
            name = info_dict[str(idx)].get("name")
            program = info_dict[str(idx)].get("program")
            is_drum = info_dict[str(idx)].get("is_drum")
            pianoroll = reconstruct_sparse(loaded, "pianoroll_" + str(idx))
            if pianoroll.dtype == np.bool_:
                track: Track = BinaryTrack(
                    name=name,
                    program=program,
                    is_drum=is_drum,
                    pianoroll=pianoroll,
                )
            elif pianoroll.dtype == np.uint8:
                track = StandardTrack(
                    name=name,
                    program=program,
                    is_drum=is_drum,
                    pianoroll=pianoroll,
                )
            else:
                track = Track(
                    name=name,
                    program=program,
                    is_drum=is_drum,
                    pianoroll=pianoroll,
                )
            tracks.append(track)
            idx += 1

        return Multitrack(
            name=info_dict["name"],
            resolution=resolution,
            tempo=loaded.get("tempo"),
            downbeat=loaded.get("downbeat"),
            tracks=tracks,
        )


def from_pretty_midi(
    midi: PrettyMIDI,
    resolution: int = DEFAULT_RESOLUTION,
    mode: str = "max",
    algorithm: str = "normal",
    collect_onsets_only: bool = False,
    first_beat_time: Optional[float] = None,
) -> Multitrack:
    """Return a Multitrack object converted from a PrettyMIDI object.

    Parse a :class:`pretty_midi.PrettyMIDI` object. The data type of the
    resulting piano rolls is automatically determined (int if 'mode' is
    'sum' and np.uint8 if `mode` is 'max').

    Parameters
    ----------
    midi : :class:`pretty_midi.PrettyMIDI`
        PrettyMIDI object to parse.
    mode : {'max', 'sum'}
        Merging strategy for duplicate notes. Defaults to 'max'.
    algorithm : {'normal', 'strict', 'custom'}
        Algorithm for finding the location of the first beat (see
        Notes). Defaults to 'normal'.
    collect_onsets_only : bool
        True to collect only the onset of the notes (i.e. note on
        events) in all tracks, where the note off and duration
        information are discarded. False to parse regular piano rolls.
        Defaults to False.
    first_beat_time : float, optional
        Location of the first beat, in sec. Required and only
        effective when using 'custom' algorithm.

    Returns
    -------
    :class:`pypianoroll.Multitrack`
        Converted Multitrack object.

    Notes
    -----
    There are three algorithms for finding the location of the first
    beat:

    - 'normal' : Estimate the location of the first beat using
      :meth:`pretty_midi.PrettyMIDI.estimate_beat_start`.
    - 'strict' : Set the location of the first beat to the time of the
      first time signature change. Raise a RuntimeError if no time
      signature change is found.
    - 'custom' : Set the location of the first beat to the value of
      argument `first_beat_time`. Raise a ValueError if
      `first_beat_time` is not given.

    If an incomplete beat before the first beat is found, an additional
    beat will be added before the (estimated) beat starting time.
    However, notes before the (estimated) beat starting time for more
    than one beat are dropped.

    """
    if mode not in ("max", "sum"):
        raise ValueError("`mode` must be either 'max' or 'sum'.")

    # Set first_beat_time for 'normal' and 'strict' modes
    if algorithm == "normal":
        if midi.time_signature_changes:
            midi.time_signature_changes.sort(key=lambda x: x.time)
            first_beat_time = midi.time_signature_changes[0].time
        else:
            first_beat_time = midi.estimate_beat_start()
    elif algorithm == "strict":
        if not midi.time_signature_changes:
            raise RuntimeError(
                "No time signature change event found. Unable to set beat "
                "start time using 'strict' algorithm."
            )
        midi.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = midi.time_signature_changes[0].time
    elif algorithm == "custom":
        if first_beat_time is None:
            raise TypeError(
                "`first_beat_time` must be given when using 'custom' "
                "algorithm."
            )
        if first_beat_time < 0.0:
            raise ValueError("`first_beat_time` must be a positive number.")
    else:
        raise ValueError(
            "`algorithm` must be one of 'normal', 'strict' or 'custom'."
        )

    # get tempo change event times and contents
    tc_times, tempi = midi.get_tempo_changes()
    arg_sorted = np.argsort(tc_times)
    tc_times = tc_times[arg_sorted]
    tempi = tempi[arg_sorted]

    beat_times = midi.get_beats(first_beat_time)
    if not beat_times.size:
        raise ValueError("Cannot get beat timings to quantize the piano roll.")
    beat_times.sort()

    n_beats = len(beat_times)
    n_time_steps = resolution * n_beats

    # Parse downbeat array
    if not midi.time_signature_changes:
        downbeat = None
    else:
        downbeat = np.zeros((n_time_steps, 1), bool)
        downbeat[0] = True
        start = 0
        end = start
        for idx, tsc in enumerate(midi.time_signature_changes[:-1]):
            end += np.searchsorted(
                beat_times[end:], midi.time_signature_changes[idx + 1].time
            )
            start_idx = start * resolution
            end_idx = end * resolution
            stride = tsc.numerator * resolution
            downbeat[start_idx:end_idx:stride] = True
            start = end

    # Build tempo array
    one_more_beat = 2 * beat_times[-1] - beat_times[-2]
    beat_times_one_more = np.append(beat_times, one_more_beat)
    bpm = 60.0 / np.diff(beat_times_one_more)
    tempo = np.tile(bpm, (1, 24)).reshape(-1, 1)

    # Parse the tracks
    tracks = []
    for instrument in midi.instruments:
        if mode == "max":
            pianoroll = np.zeros((n_time_steps, 128), np.uint8)
        else:
            pianoroll = np.zeros((n_time_steps, 128), int)

        pitches = np.array(
            [
                note.pitch
                for note in instrument.notes
                if note.end > first_beat_time
            ]
        )
        note_on_times = np.array(
            [
                note.start
                for note in instrument.notes
                if note.end > first_beat_time
            ]
        )
        beat_indices = np.searchsorted(beat_times, note_on_times) - 1
        remained = note_on_times - beat_times[beat_indices]
        ratios = remained / (
            beat_times_one_more[beat_indices + 1] - beat_times[beat_indices]
        )
        rounded = np.round((beat_indices + ratios) * resolution)
        note_ons = rounded.astype(int)

        if collect_onsets_only:
            pianoroll[note_ons, pitches] = True
        elif instrument.is_drum:
            velocities = [
                note.velocity
                for note in instrument.notes
                if note.end > first_beat_time
            ]
            pianoroll[note_ons, pitches] = velocities
        else:
            note_off_times = np.array(
                [
                    note.end
                    for note in instrument.notes
                    if note.end > first_beat_time
                ]
            )
            beat_indices = np.searchsorted(beat_times, note_off_times) - 1
            remained = note_off_times - beat_times[beat_indices]
            ratios = remained / (
                beat_times_one_more[beat_indices + 1]
                - beat_times[beat_indices]
            )
            note_offs = ((beat_indices + ratios) * resolution).astype(int)

            for idx, start in enumerate(note_ons):
                end = note_offs[idx]
                velocity = instrument.notes[idx].velocity

                if velocity < 1:
                    continue

                if 0 < start < n_time_steps:
                    if pianoroll[start - 1, pitches[idx]]:
                        pianoroll[start - 1, pitches[idx]] = 0
                if end < n_time_steps - 1:
                    if pianoroll[end, pitches[idx]]:
                        end -= 1

                if mode == "max":
                    pianoroll[start:end, pitches[idx]] += velocity
                else:
                    maximum = np.maximum(
                        pianoroll[start:end, pitches[idx]], velocity
                    )
                    pianoroll[start:end, pitches[idx]] = maximum

        if mode == "max":
            track: Track = StandardTrack(
                name=str(instrument.name),
                program=int(instrument.program),
                is_drum=bool(instrument.is_drum),
                pianoroll=pianoroll,
            )
        else:
            track = Track(
                name=str(instrument.name),
                program=int(instrument.program),
                is_drum=bool(instrument.is_drum),
                pianoroll=pianoroll,
            )
        tracks.append(track)

    return Multitrack(
        resolution=resolution, tempo=tempo, downbeat=downbeat, tracks=tracks
    )


def read(path: Union[str, Path], **kwargs) -> Multitrack:
    """Read a MIDI file into a Multitrack object.

    Parameters
    ----------
    path : str or Path
        Path to the file to read.
    **kwargs
        Keyword arguments to pass to
        :func:`pypianoroll.from_pretty_midi`.

    See Also
    --------
    :func:`pypianoroll.write` : Write a Multitrack object to a MIDI
      file.
    :func:`pypianoroll.load` : Load a NPZ file into a Multitrack object.

    """
    return from_pretty_midi(PrettyMIDI(str(path)), **kwargs)

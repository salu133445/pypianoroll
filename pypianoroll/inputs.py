"""MIDI I/O interface."""
import json
from typing import Optional

import numpy as np
from pretty_midi import PrettyMIDI

from .multitrack import DEFAULT_RESOLUTION, Multitrack, reconstruct_sparse
from .track import Track

__all__ = ["load", "from_pretty_midi", "read"]


def load(path: str):
    """Load a NPZ file into a Multitrack object.

    Supports only files previously saved by :func:`pypianoroll.save`.

    Parameters
    ----------
    path : str
        Path to the NPZ file to load.

    """
    with np.load(path) as loaded:
        if "info.json" not in loaded:
            raise ValueError("Cannot find 'info.json' in the npz file.")

        # Load the info dictionary
        info_dict = json.loads(loaded["info.json"].decode("utf-8"))

        # Get the name and resolution
        name = info_dict["name"]
        resolution = info_dict["resolution"]

        # Load the tempo and downbeat array
        tempo = loaded["tempo"] if "tempo" in loaded.files else None
        downbeat = loaded["downbeat"] if "downbeat" in loaded.files else None

        # Load the tracks
        idx = 0
        tracks = []
        while str(idx) in info_dict:
            pianoroll = reconstruct_sparse(loaded, "pianoroll_" + str(idx))
            track = Track(
                program=info_dict[str(idx)]["program"],
                is_drum=info_dict[str(idx)]["is_drum"],
                name=info_dict[str(idx)]["name"],
                pianoroll=pianoroll,
            )
            tracks.append(track)
            idx += 1

    return Multitrack(
        resolution=resolution,
        tempo=tempo,
        downbeat=downbeat,
        name=name,
        tracks=tracks,
    )


def from_pretty_midi(
    pm: PrettyMIDI,
    resolution: int = DEFAULT_RESOLUTION,
    mode: str = "max",
    algorithm: str = "normal",
    binarized: bool = False,
    skip_empty_tracks: bool = True,
    collect_onsets_only: bool = False,
    threshold: float = 0,
    first_beat_time: Optional[float] = None,
):
    """Return a Multitrack object converted from a PrettyMIDI object.

    Parse a :class:`pretty_midi.PrettyMIDI` object. The data type of the
    resulting piano rolls is automatically determined (int if 'mode' is
    'sum', np.uint8 if `mode` is 'max' and `binarized` is False, bool if
    `mode` is 'max' and `binarized` is True).

    Parameters
    ----------
    pm : `pretty_midi.PrettyMIDI` object
        A :class:`pretty_midi.PrettyMIDI` object to be parsed.
    mode : {'max', 'sum'}
        A string that indicates the merging strategy to apply to
        duplicate notes. Defaults to 'max'.
    algorithm : {'normal', 'strict', 'custom'}
        A string that indicates the method used to get the location of
        the first beat. Notes before it will be dropped unless an
        incomplete beat before it is found (see Notes for more
        information). Defaults to 'normal'.

        - The 'normal' algorithm estimates the location of the first
          beat by :meth:`pretty_midi.PrettyMIDI.estimate_beat_start`.
        - The 'strict' algorithm sets the first beat at the event time
          of the first time signature change. Raise a ValueError if no
          time signature change event is found.
        - The 'custom' algorithm takes argument `first_beat_time` as the
          location of the first beat.

    binarized : bool
        True to binarize the parsed piano rolls before merging duplicate
        notes. False to use the original parsed piano rolls. Defaults to
        False.
    skip_empty_tracks : bool
        True to remove tracks with empty piano rolls and compress the
        pitch range of the parsed piano rolls. False to retain the empty
        tracks and use the original parsed piano rolls. Deafaults to
        True.
    collect_onsets_only : bool
        True to collect only the onset of the notes (i.e. note on
        events) in all tracks, where the note off and duration
        information are dropped. False to parse regular piano rolls.
        Defaults to False.
    threshold : int or float
        A threshold used to binarize the parsed piano rolls. Only
        effective when `binarized` is True. Defaults to zero.
    first_beat_time : float
        Location of the first beat, in sec. Required and only
        effective  when using 'custom' algorithm.

    Notes
    -----
    If an incomplete beat before the first beat is found, an additional
    beat will be added before the (estimated) beat starting time.
    However, notes before the (estimated) beat starting time for more
    than one beat are dropped.

    Returns
    -------
    :class:`pypianoroll.Multitrack`
        Converted multitrack.

    """
    if mode not in ("max", "sum"):
        raise ValueError("`mode` must be one of {'max', 'sum'}.")
    if algorithm not in ("strict", "normal", "custom"):
        raise ValueError(
            "`algorithm` must be one of {'normal', 'strict', 'custom'}."
        )
    if algorithm == "custom":
        if not isinstance(first_beat_time, (int, float)):
            raise TypeError(
                "`first_beat_time` must be int or float when "
                "using 'custom' algorithm."
            )
        if first_beat_time < 0.0:
            raise ValueError(
                "`first_beat_time` must be a positive number "
                "when using 'custom' algorithm."
            )

    # Set first_beat_time for 'normal' and 'strict' modes
    if algorithm == "normal":
        if pm.time_signature_changes:
            pm.time_signature_changes.sort(key=lambda x: x.time)
            first_beat_time = pm.time_signature_changes[0].time
        else:
            first_beat_time = pm.estimate_beat_start()
    elif algorithm == "strict":
        if not pm.time_signature_changes:
            raise ValueError(
                "No time signature change event found. Unable to set beat "
                "start time using 'strict' algorithm."
            )
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time

    # get tempo change event times and contents
    tc_times, tempi = pm.get_tempo_changes()
    arg_sorted = np.argsort(tc_times)
    tc_times = tc_times[arg_sorted]
    tempi = tempi[arg_sorted]

    beat_times = pm.get_beats(first_beat_time)
    if not beat_times.size:
        raise ValueError("Cannot get beat timings to quantize the piano roll.")
    beat_times.sort()

    n_beats = len(beat_times)
    n_time_steps = resolution * n_beats

    # Parse downbeat array
    if not pm.time_signature_changes:
        downbeat = None
    else:
        downbeat = np.zeros((n_time_steps, 1), bool)
        downbeat[0] = True
        start = 0
        end = start
        for idx, tsc in enumerate(pm.time_signature_changes[:-1]):
            end += np.searchsorted(
                beat_times[end:], pm.time_signature_changes[idx + 1].time
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
    for instrument in pm.instruments:
        if binarized:
            pianoroll = np.zeros((n_time_steps, 128), bool)
        elif mode == "max":
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
            if binarized:
                pianoroll[note_ons, pitches] = True
            else:
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
                if binarized and velocity <= threshold:
                    continue

                if 0 < start < n_time_steps:
                    if pianoroll[start - 1, pitches[idx]]:
                        pianoroll[start - 1, pitches[idx]] = 0
                if end < n_time_steps - 1:
                    if pianoroll[end, pitches[idx]]:
                        end -= 1

                if binarized:
                    if mode == "sum":
                        pianoroll[start:end, pitches[idx]] += 1
                    elif mode == "max":
                        pianoroll[start:end, pitches[idx]] = True
                elif mode == "sum":
                    pianoroll[start:end, pitches[idx]] += velocity
                elif mode == "max":
                    maximum = np.maximum(
                        pianoroll[start:end, pitches[idx]], velocity
                    )
                    pianoroll[start:end, pitches[idx]] = maximum

        if skip_empty_tracks and not np.any(pianoroll):
            continue

        track = Track(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
            pianoroll=pianoroll,
        )
        tracks.append(track)

    return Multitrack(
        resolution=resolution, tempo=tempo, downbeat=downbeat, tracks=tracks
    )


def read(path: str, **kwargs):
    """Read a MIDI file into a Multitrack object.

    See :meth:`pypianoroll.from_pretty_midi` for full documentation.

    Parameters
    ----------
    path : str
        Path to the MIDI file to read.

    """
    pm = PrettyMIDI(path)
    return from_pretty_midi(pm, **kwargs)

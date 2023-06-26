"""Output interfaces.

Functions
---------

- save
- to_pretty_midi
- write

Variable
--------

- DEFAULT_TEMPO
- DEFAULT_VELOCITY

"""
import json
import zipfile
from fractions import Fraction
from copy import deepcopy
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import numpy as np
import pretty_midi
from pretty_midi import Instrument, PrettyMIDI

from .track import BinaryTrack, StandardTrack
from .utils import decompose_sparse

if TYPE_CHECKING:
    from .multitrack import Multitrack

__all__ = [
    "save",
    "to_pretty_midi",
    "write",
    "DEFAULT_TEMPO",
    "DEFAULT_VELOCITY",
]

DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 64


def save(
    path: Union[str, Path], multitrack: "Multitrack", compressed: bool = True
):
    """Save a Multitrack object to a NPZ file.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ file to save.
    multitrack : :class:`pypianoroll.Multitrack`
        Multitrack to save.
    compressed : bool, default: True
        Whether to save to a compressed NPZ file.

    Notes
    -----
    To reduce the file size, the piano rolls are first converted to
    instances of :class:`scipy.sparse.csc_matrix`. The component
    arrays are then collected and saved to a npz file.

    See Also
    --------
    :func:`pypianoroll.load` : Load a NPZ file into a Multitrack object.
    :func:`pypianoroll.write` : Write a Multitrack object to a MIDI
      file.

    """
    info_dict: Dict = {
        "resolution": multitrack.resolution,
        "name": multitrack.name,
    }

    array_dict = {}
    if multitrack.tempo is not None:
        array_dict["tempo"] = multitrack.tempo
    if multitrack.beat is not None:
        array_dict["beat"] = multitrack.beat
    if multitrack.downbeat is not None:
        array_dict["downbeat"] = multitrack.downbeat

    for idx, track in enumerate(multitrack.tracks):
        array_dict.update(
            decompose_sparse(track.pianoroll, "pianoroll_" + str(idx))
        )
        info_dict[str(idx)] = {
            "program": track.program,
            "is_drum": track.is_drum,
            "name": track.name,
        }

    if compressed:
        np.savez_compressed(path, **array_dict)
    else:
        np.savez(path, **array_dict)

    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "a") as zip_file:
        zip_file.writestr("info.json", json.dumps(info_dict), compression)


def to_pretty_midi(
    multitrack: "Multitrack",
    default_tempo: float = None,
    default_velocity: int = DEFAULT_VELOCITY,
) -> PrettyMIDI:
    """Return a Multitrack object as a PrettyMIDI object.

    Parameters
    ----------
    default_tempo : int, default: `pypianoroll.DEFAULT_TEMPO` (120)
        Default tempo to use. If attribute `tempo` is available, encorporate
        it into the midi file with any tempo changes and time signature changes that occur
    default_velocity : int, default: `pypianoroll.DEFAULT_VELOCITY` (64)
        Default velocity to assign to binarized tracks.

    Returns
    -------
    :class:`pretty_midi.PrettyMIDI`
        Converted PrettyMIDI object.

    Notes
    -----
    - Time signature changes default to */4.
    - The velocities of the converted piano rolls will be clipped to
      [0, 127].
    - Adjacent nonzero values of the same pitch will be considered
      a single note with their mean as its velocity.

    """
    if default_tempo is not None:
        tempo = np.full(multitrack.get_max_length(), default_tempo)
    elif multitrack.tempo is not None:
        tempo = multitrack.tempo[:,0]
    else:
        tempo = np.full(multitrack.get_max_length(), DEFAULT_TEMPO)

    # Create a PrettyMIDI instance
    midi = PrettyMIDI(initial_tempo=tempo[0])

    # Compute length of a time step
    time_step_length = 60.0 / tempo / multitrack.resolution
    # Use prefix sum for fast computation of elapsed time
    prefix = np.concatenate(((0,), np.add.accumulate(time_step_length)))

    # Find tempi changes
    tempi_changes = np.diff(tempo).nonzero()[0] + 1

    # Copied from pretty-midi source code (https://github.com/craffel/pretty-midi/blob/241279b91196125881724e53ea436e1b9181f74b/pretty_midi/pretty_midi.py)
    # Changes tempo by updating ticks
    last_tick, last_tick_scale = midi._tick_scales[0]
    previous_time = 0.
    for time, tempo in zip(prefix[tempi_changes], tempo[tempi_changes]):
        # Compute new tick location as the last tick plus the time between
        # the last and next tempo change, scaled by the tick scaling
        tick = last_tick + (time - previous_time)/last_tick_scale
        # Update the tick scale
        tick_scale = 60.0/(tempo*midi.resolution)
        # Don't add tick scales if they are repeats
        if tick_scale != last_tick_scale:
            # Add in the new tick scale
            midi._tick_scales.append((int(round(tick)), tick_scale))
            # Update the time and values of the previous tick scale
            previous_time = time
            last_tick, last_tick_scale = tick, tick_scale
    midi._update_tick_to_time(midi._tick_scales[-1][0] + 1)

    for track in multitrack.tracks:
        instrument = Instrument(
            program=track.program, is_drum=track.is_drum, name=track.name
        )
        if isinstance(track, BinaryTrack):
            processed = track.set_nonzeros(default_velocity)
        elif isinstance(track, StandardTrack):
            copied = deepcopy(track)
            processed = copied.clip()
        else:
            raise ValueError(
                f"Expect BinaryTrack or StandardTrack, but got {type(track)}."
            )
        clipped = processed.pianoroll.astype(np.uint8)
        binarized = clipped > 0
        padded = np.pad(binarized, ((1, 1), (0, 0)), "constant")
        diff = np.diff(padded.astype(np.int8), axis=0)

        positives = np.nonzero((diff > 0).T)
        pitches = positives[0]
        note_ons = positives[1]
        note_on_times = prefix[note_ons]
        note_offs = np.nonzero((diff < 0).T)[1]
        note_off_times = prefix[note_offs]

        for idx, pitch in enumerate(pitches):
            velocity = np.mean(clipped[note_ons[idx] : note_offs[idx], pitch])
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=pitch,
                start=note_on_times[idx],
                end=note_off_times[idx],
            )
            instrument.notes.append(note)

        instrument.notes.sort(key=attrgetter("start"))
        midi.instruments.append(instrument)

    if multitrack.downbeat is not None:
        # Find downbeat positions
        downbeats = np.concatenate((multitrack.downbeat[:,0], (True,)))
        indices = np.where(downbeats)[0]

        for i in range(len(indices) - 1):
            # Calculate number of beats
            beats = Fraction((indices[i+1] - indices[i]) / multitrack.resolution / 4).limit_denominator(16)
            time = prefix[indices[i]] if i != 0 else 0 # include timesignature at time 0
            # Find the first denominator that fits into the beat without 1/*
            beat_denominators = [4, 8, 16]
            for denominator in beat_denominators:
                if denominator % beats.denominator == 0 and (denominator == beat_denominators[-1] or beats.numerator * denominator // beats.denominator != 1):
                    midi.time_signature_changes.append(pretty_midi.TimeSignature(beats.numerator * denominator // beats.denominator, denominator, time))
                    break

    return midi


def write(path: str, multitrack: "Multitrack"):
    """Write a Multitrack object to a MIDI file.

    Parameters
    ----------
    path : str
        Path to write the file.
    multitrack : :class:`pypianoroll.Multitrack`
        Multitrack to save.

    See Also
    --------
    :func:`pypianoroll.read` : Read a MIDI file into a Multitrack
      object.
    :func:`pypianoroll.save` : Save a Multitrack object to a NPZ file.

    """
    return to_pretty_midi(multitrack).write(str(path))

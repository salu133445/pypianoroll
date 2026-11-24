"""Test cases for Multitrack class."""
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

import pypianoroll

from .utils import multitrack


def test_save_load(multitrack, tmp_path):
    multitrack.save(tmp_path / "test.npz")
    loaded = pypianoroll.load(tmp_path / "test.npz")
    assert np.allclose(loaded.downbeat, multitrack.downbeat)
    assert loaded.resolution == multitrack.resolution
    assert loaded.name == multitrack.name
    assert np.allclose(
        loaded.tracks[0].pianoroll, multitrack.tracks[0].pianoroll
    )
    assert loaded.tracks[0].program == 0
    assert not loaded.tracks[0].is_drum
    assert loaded.tracks[0].name == "track_1"
    assert np.allclose(
        loaded.tracks[1].pianoroll, multitrack.tracks[1].pianoroll
    )
    assert loaded.tracks[1].program == 0
    assert loaded.tracks[1].is_drum
    assert loaded.tracks[1].name == "track_2"


def test_write_read(multitrack, tmp_path):
    multitrack.write(tmp_path / "test.mid")
    loaded = pypianoroll.read(tmp_path / "test.mid")
    assert np.allclose(
        loaded.tracks[0].pianoroll, multitrack.tracks[0].pianoroll
    )
    assert loaded.tracks[0].program == 0
    assert not loaded.tracks[0].is_drum
    assert loaded.tracks[0].name == "track_1"
    assert np.allclose(
        (loaded.tracks[1].pianoroll > 0), multitrack.tracks[1].pianoroll,
    )
    assert loaded.tracks[1].program == 0
    assert loaded.tracks[1].is_drum
    assert loaded.tracks[1].name == "track_2"


def test_read_realworld(tmp_path):
    loaded = pypianoroll.read(Path(__file__).parent / "fur-elise.mid")
    loaded.write(tmp_path / "test.mid")


@pytest.mark.parametrize(
    "mode, expected", [("max", [100, 100, 90]), ("sum", [270, 190, 90])]
)
def test_from_pretty_midi_overlapping_velocities(mode, expected):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    piano.notes = [
        pretty_midi.Note(velocity=80, pitch=60, start=0, end=0.5),
        pretty_midi.Note(velocity=100, pitch=60, start=0, end=1.0),
        pretty_midi.Note(velocity=90, pitch=60, start=0, end=1.5),
    ]
    midi.instruments.append(piano)

    roll = pypianoroll.from_pretty_midi(midi, resolution=1, mode=mode)
    np.testing.assert_array_equal(roll.tracks[0].pianoroll[:, 60], expected)

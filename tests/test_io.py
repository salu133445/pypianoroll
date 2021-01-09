"""Test cases for Multitrack class."""
from pathlib import Path

import numpy as np

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

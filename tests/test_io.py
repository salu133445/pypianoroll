"""Test cases for Multitrack class."""
import numpy as np
from pytest import fixture

import pypianoroll
from pypianoroll import BinaryTrack, Multitrack, StandardTrack

from .utils import multitrack


def test_save_load(multitrack, tmp_path):
    path = tmp_path / "test.npz"
    multitrack.save(path)
    loaded = pypianoroll.load(path)
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
    path = tmp_path / "test.mid"
    multitrack.write(path)
    loaded = pypianoroll.read(path)
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

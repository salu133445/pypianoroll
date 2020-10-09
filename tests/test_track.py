"""Test cases for Track class."""
import numpy as np
from pytest import fixture

from pypianoroll import Track


@fixture
def track():
    pianoroll = np.zeros((96, 128), np.uint8)
    pianoroll[:95, [60, 64, 67, 72, 76, 79, 84]] = 100
    return Track(program=0, is_drum=False, name="test", pianoroll=pianoroll)


def test_slice(track):
    sliced = track[20:40]
    assert isinstance(sliced, Track)
    assert sliced.pianoroll.shape == (20, 128)


def test_assign_constant(track):
    track.assign_constant(50)
    assert track.pianoroll[0, 60] == 50


def test_binarize(track):
    track.binarize()
    assert np.issubdtype(track.pianoroll.dtype, np.bool_)
    assert track.pianoroll[0, 60]
    assert not track.pianoroll[0, 0]


def test_is_binarized(track):
    assert not track.is_binarized()


def test_clip(track):
    track.clip(10, 50)
    assert track.pianoroll[0, 0] == 10
    assert track.pianoroll[0, 60] == 50


def test_pad(track):
    track.pad(10)
    assert track.pianoroll.shape == (106, 128)


def test_pad_to_multiple(track):
    track.pad_to_multiple(25)
    assert track.pianoroll.shape == (100, 128)


def test_get_active_length(track):
    assert track.get_active_length() == 95


def test_get_active_pitch_range(track):
    assert track.get_active_pitch_range() == (60, 84)


def test_transpose(track):
    track.transpose(5)
    assert track.pianoroll[0, 65] == 100


def trim_trailing_silence(track):
    track.trim_trailing_silence()
    assert track.pianoroll[0] == 95


@fixture
def binary_track():
    pianoroll = np.zeros((96, 128), bool)
    pianoroll[:95, [60, 64, 67, 72]] = True
    return Track(program=0, is_drum=False, name="test", pianoroll=pianoroll)


def test_assign_constant_binary_track(binary_track):
    binary_track.assign_constant(50.0)
    assert np.issubdtype(binary_track.pianoroll.dtype, np.floating)
    assert binary_track.pianoroll[0, 60] == 50.0


def test_is_binarized_binary_track(binary_track):
    assert binary_track.is_binarized()

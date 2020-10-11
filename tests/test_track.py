"""Test cases for Track class."""
import numpy as np
from pytest import fixture

from pypianoroll import BinaryTrack, StandardTrack


@fixture
def track():
    pianoroll = np.zeros((96, 128), np.uint8)
    pianoroll[:95, [60, 64, 67, 72, 76, 79, 84]] = 100
    return StandardTrack(
        program=0, is_drum=False, name="test", pianoroll=pianoroll
    )


def test_repr(track):
    assert repr(track) == (
        "StandardTrack(name='test', program=0, is_drum=False, "
        "pianoroll=array(shape=(96, 128)))"
    )


def test_slice(track):
    sliced = track[20:40]
    assert isinstance(sliced, StandardTrack)
    assert sliced.pianoroll.shape == (20, 128)


def test_assign_constant(track):
    track.assign_constant(50)
    assert track.pianoroll[0, 60] == 50


def test_binarize(track):
    binarized = track.binarize()
    assert binarized.pianoroll.dtype == np.bool_
    assert binarized.pianoroll[0, 60]
    assert not binarized.pianoroll[0, 0]


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
    return BinaryTrack(
        program=0, is_drum=True, name="test", pianoroll=pianoroll
    )


def test_repr_binary(binary_track):
    assert repr(binary_track) == (
        "BinaryTrack(program=0, is_drum=True, name='test', "
        "pianoroll=array(shape=(96, 128)))"
    )


def test_slice_binary(binary_track):
    sliced = binary_track[20:40]
    assert isinstance(sliced, BinaryTrack)
    assert sliced.pianoroll.shape == (20, 128)


def test_assign_constant_binary(binary_track):
    track = binary_track.assign_constant(50)
    assert track.pianoroll.dtype == np.uint8
    assert track.pianoroll[0, 60] == 50

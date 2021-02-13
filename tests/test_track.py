"""Test cases for Track class."""
import numpy as np
from numpy import ndarray
from pytest import fixture

from pypianoroll import BinaryTrack, StandardTrack, Track


def test_empty_track():
    track = Track("test", 0, False)
    assert track.is_valid()
    assert not track.pianoroll.size


def test_constructor_standard():
    track = StandardTrack("test", 0, False, np.zeros((96, 128), float))
    assert track.pianoroll.dtype == np.uint8


def test_constructor_binary():
    track = BinaryTrack("test", 0, False, np.zeros((96, 128), float))
    assert track.pianoroll.dtype == np.bool_


@fixture
def track():
    pianoroll = np.zeros((96, 128), np.float32)
    pianoroll[:95, [60, 64, 67, 72, 76, 79, 84]] = 100
    return Track("test", 0, False, pianoroll)


def test_repr(track):
    assert repr(track) == (
        "Track(name='test', program=0, is_drum=False, "
        "pianoroll=array(shape=(96, 128), dtype=float32))"
    )


def test_len(track):
    assert len(track) == 96


def test_slice(track):
    sliced = track[20:40]
    assert isinstance(sliced, ndarray)
    assert sliced.shape == (20, 128)


def test_is_valid(track):
    assert track.is_valid()


def test_get_length(track):
    assert track.get_length() == 95


def test_copy(track):
    copied = track.copy()
    assert id(copied) != id(track)
    assert id(copied.pianoroll) != id(track.pianoroll)


def test_pad(track):
    track.pad(10)
    assert track.pianoroll.shape == (106, 128)


def test_pad_to_multiple(track):
    track.pad_to_multiple(25)
    assert track.pianoroll.shape == (100, 128)


def test_transpose(track):
    track.transpose(5)
    assert track.pianoroll[0, 65] == 100


def test_trim(track):
    track.trim(20, 40)
    assert isinstance(track, Track)
    assert track.pianoroll.shape == (20, 128)


def test_trim_auto(track):
    track.trim()
    assert isinstance(track, Track)
    assert track.pianoroll.shape == (95, 128)


def test_standardize(track):
    standardized = track.standardize()
    assert isinstance(standardized, StandardTrack)
    assert standardized.is_valid()
    assert standardized.pianoroll.dtype == np.uint8


@fixture
def standard_track():
    pianoroll = np.zeros((96, 128), np.uint8)
    pianoroll[:95, [60, 64, 67, 72, 76, 79, 84]] = 100
    return StandardTrack("test", 0, False, pianoroll)


def test_repr_standard(standard_track):
    assert repr(standard_track) == (
        "StandardTrack(name='test', program=0, is_drum=False, "
        "pianoroll=array(shape=(96, 128)))"
    )


def test_set_nonzeros(standard_track):
    standard_track.set_nonzeros(50)
    assert standard_track.pianoroll[0, 60] == 50


def test_clip(standard_track):
    standard_track.clip(10, 50)
    assert standard_track.pianoroll[0, 0] == 10
    assert standard_track.pianoroll[0, 60] == 50


def test_binarize(standard_track):
    binarized = standard_track.binarize()
    assert isinstance(binarized, BinaryTrack)
    assert binarized.pianoroll.dtype == np.bool_
    assert binarized.pianoroll[0, 60]
    assert not binarized.pianoroll[0, 0]


@fixture
def binary_track():
    pianoroll = np.zeros((96, 128), bool)
    pianoroll[:95, [60, 64, 67, 72]] = True
    return BinaryTrack(
        name="test", program=0, is_drum=True, pianoroll=pianoroll
    )


def test_repr_binary(binary_track):
    assert repr(binary_track) == (
        "BinaryTrack(name='test', program=0, is_drum=True, "
        "pianoroll=array(shape=(96, 128)))"
    )


def test_slice_binary(binary_track):
    sliced = binary_track[20:40]
    assert isinstance(sliced, ndarray)
    assert sliced.shape == (20, 128)


def test_set_nonzeros_binary(binary_track):
    track = binary_track.set_nonzeros(50)
    assert isinstance(track, StandardTrack)
    assert track.pianoroll.dtype == np.uint8
    assert track.pianoroll[0, 60] == 50

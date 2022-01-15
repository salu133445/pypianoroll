"""Test cases for Multitrack class."""
import numpy as np
from pytest import fixture

from pypianoroll import BinaryTrack, Multitrack, StandardTrack

from .utils import multitrack


def test_repr(multitrack):
    assert repr(multitrack) == (
        "Multitrack(name='test', resolution=24, "
        "beat=array(shape=(96,), dtype=bool), "
        "downbeat=array(shape=(96,), dtype=bool), tracks=["
        "StandardTrack(name='track_1', program=0, is_drum=False, "
        "pianoroll=array(shape=(96, 128), dtype=uint8)), "
        "BinaryTrack(name='track_2', program=0, is_drum=True, "
        "pianoroll=array(shape=(96, 128), dtype=bool))])"
    )


def test_len(multitrack):
    assert len(multitrack) == 2


def test_slice(multitrack):
    sliced = multitrack[1]
    assert isinstance(sliced, BinaryTrack)
    assert sliced.pianoroll.shape == (96, 128)


def test_is_valid(multitrack):
    multitrack.validate()


def test_get_length(multitrack):
    assert multitrack.get_length() == 95


def test_get_beat_steps(multitrack):
    assert np.all(multitrack.get_beat_steps() == [0, 24, 48, 72])


def test_get_downbeat_steps(multitrack):
    assert np.all(multitrack.get_downbeat_steps() == [0])


def test_set_nonzeros(multitrack):
    multitrack.set_nonzeros(50)
    assert isinstance(multitrack.tracks[1], StandardTrack)
    assert multitrack.tracks[1].pianoroll[0, 36] == 50


def test_set_resolution(multitrack):
    multitrack.set_resolution(4)
    assert np.all(multitrack.get_beat_steps() == [0, 4, 8, 12])
    assert multitrack.tracks[0].pianoroll[15, 60] == 100
    assert np.all(
        multitrack.tracks[1].pianoroll[[0, 3, 5, 8, 11, 13], 36] == 1
    )


def test_copy(multitrack):
    copied = multitrack.copy()
    assert id(copied) != id(multitrack)
    assert id(copied.downbeat) != id(multitrack.downbeat)
    assert id(copied.tracks[0]) != id(multitrack.tracks[0])
    assert id(copied.tracks[0].pianoroll) != id(multitrack.tracks[0].pianoroll)


def test_count_beat(multitrack):
    assert multitrack.count_beat() == 4


def test_count_downbeat(multitrack):
    assert multitrack.count_downbeat() == 1


def test_stack(multitrack):
    stacked = multitrack.stack()
    assert stacked.shape == (2, 96, 128)


@fixture
def multitrack_to_blend():
    pianoroll_1 = np.zeros((96, 128), np.uint8)
    pianoroll_1[:95, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        name="track_1", program=0, is_drum=False, pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), np.uint8)
    pianoroll_2[:95, [60, 64, 67, 72]] = 100
    track_2 = StandardTrack(
        name="track_2", program=0, is_drum=True, pianoroll=pianoroll_2
    )
    downbeat = np.zeros((96,), bool)
    downbeat[0] = True
    return Multitrack(
        name="test",
        resolution=24,
        downbeat=downbeat,
        tracks=[track_1, track_2],
    )


def test_blend_any(multitrack_to_blend):
    blended = multitrack_to_blend.blend("any")
    assert blended.dtype == np.bool_
    assert not blended[0, 0]
    assert blended[0, 60]


def test_blend_sum(multitrack_to_blend):
    blended = multitrack_to_blend.blend("sum")
    assert blended.dtype == np.uint8
    assert blended[0, 0] == 0
    assert blended[0, 60] == 127


def test_blend_max(multitrack_to_blend):
    blended = multitrack_to_blend.blend("max")
    assert blended.dtype == np.uint8
    assert blended[0, 0] == 0
    assert blended[0, 60] == 100


def test_append(multitrack):
    pianoroll = np.zeros((96, 128), np.bool_)
    pianoroll[:95:16, 41] = True
    track_to_append = BinaryTrack(name="track_3", pianoroll=pianoroll)
    multitrack.append(track_to_append)
    assert len(multitrack.tracks) == 3
    assert multitrack.tracks[2].name == "track_3"


def test_binarize(multitrack):
    multitrack.binarize()
    assert isinstance(multitrack.tracks[0], BinaryTrack)
    assert multitrack.tracks[0].pianoroll[0, 60] == 1


def test_clip(multitrack):
    multitrack.clip(upper=60)
    assert multitrack.tracks[0].pianoroll[0, 60] == 60


def test_pad_to_same(multitrack):
    pianoroll_1 = np.zeros((96, 128), np.uint8)
    pianoroll_1[0:95, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        name="track_1", program=0, is_drum=False, pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), np.bool)
    pianoroll_2[0:95:16, 36] = True
    track_2 = BinaryTrack(
        name="track_2", program=0, is_drum=True, pianoroll=pianoroll_2
    )
    downbeat = np.zeros((96,), bool)
    downbeat[0] = True
    multitrack = Multitrack(
        name="test",
        resolution=24,
        downbeat=downbeat,
        tracks=[track_1, track_2],
    )
    multitrack.pad_to_same()
    assert multitrack.tracks[0].pianoroll.shape[0] == 96
    assert multitrack.tracks[1].pianoroll.shape[0] == 96


def test_remove_empty():
    pianoroll_1 = np.zeros((96, 128), np.uint8)
    pianoroll_1[0:95, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        name="track_1", program=0, is_drum=False, pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), np.bool)
    track_2 = StandardTrack(
        name="track_2", program=0, is_drum=True, pianoroll=pianoroll_2
    )
    downbeat = np.zeros((96,), bool)
    downbeat[0] = True
    multitrack = Multitrack(
        name="test",
        resolution=24,
        downbeat=downbeat,
        tracks=[track_1, track_2],
    )
    multitrack.remove_empty()
    assert len(multitrack) == 1


def test_trim(multitrack):
    multitrack.trim()
    assert multitrack.tracks[0].pianoroll.shape == (95, 128)
    assert multitrack.tracks[1].pianoroll.shape == (95, 128)

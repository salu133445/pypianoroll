"""Test cases for Multitrack class."""
import numpy as np
from pytest import fixture

import pypianoroll
from pypianoroll import BinaryTrack, Multitrack, StandardTrack


@fixture
def multitrack():
    pianoroll_1 = np.zeros((192, 128), np.uint8)
    pianoroll_1[:191, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        name="track_1", program=0, is_drum=False, pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((192, 128), np.bool)
    pianoroll_2[:191:16, 36] = True
    track_2 = BinaryTrack(
        name="track_2", program=0, is_drum=True, pianoroll=pianoroll_2
    )
    downbeat = np.zeros((192, 1), bool)
    downbeat[[0, 96]] = True
    return Multitrack(
        name="test",
        resolution=24,
        downbeat=downbeat,
        tracks=[track_1, track_2],
    )


def test_repr(multitrack):
    assert repr(multitrack) == (
        "Multitrack(name='test', resolution=24, tempo=None, "
        "downbeat=array(shape=(192, 1)), tracks=["
        "StandardTrack(name='track_1', program=0, is_drum=False, "
        "pianoroll=array(shape=(192, 128))), "
        "BinaryTrack(name='track_2', program=0, is_drum=True, "
        "pianoroll=array(shape=(192, 128)))])"
    )


def test_slice(multitrack):
    sliced = multitrack[1, 20:40]
    assert isinstance(sliced, Multitrack)
    assert len(sliced.tracks) == 1
    assert sliced.tracks[0].pianoroll.shape == (20, 128)


def test_append_track(multitrack):
    pianoroll = np.zeros((192, 128), np.bool_)
    pianoroll[:191:16, 41] = True
    track_to_append = BinaryTrack(name="track_3", pianoroll=pianoroll)
    multitrack.append(track_to_append)
    assert len(multitrack.tracks) == 3
    assert multitrack.tracks[2].name == "track_3"


def test_get_active_length(multitrack):
    assert multitrack.get_active_length() == 191


def test_get_active_pitch_range(multitrack):
    assert multitrack.get_active_pitch_range() == (36, 72)


def test_get_downbeat_steps(multitrack):
    assert np.all(multitrack.get_downbeat_steps() == [0, 96])


def test_get_max_length(multitrack):
    assert multitrack.get_max_length() == 192


def test_count_downbeat(multitrack):
    assert multitrack.count_downbeat() == 2


def test_stack(multitrack):
    stacked = multitrack.stack()
    assert stacked.shape == (2, 192, 128)


def test_trim_trailing_silence(multitrack):
    multitrack.trim_trailing_silence()
    assert multitrack.tracks[0].pianoroll.shape[0] == 191
    assert multitrack.tracks[1].pianoroll.shape[0] == 191


def test_pad_to_same(multitrack):
    pianoroll_1 = np.zeros((192, 128), np.uint8)
    pianoroll_1[0:191, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        program=0, is_drum=False, name="track_1", pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), np.bool)
    pianoroll_2[0:95:16, 36] = True
    track_2 = BinaryTrack(
        program=0, is_drum=True, name="track_2", pianoroll=pianoroll_2
    )
    downbeat = np.zeros((192, 1), bool)
    downbeat[[0, 96]] = True
    multitrack = Multitrack(
        resolution=24, downbeat=downbeat, tracks=[track_1, track_2]
    )
    multitrack.pad_to_same()
    assert multitrack.tracks[0].pianoroll.shape[0] == 192
    assert multitrack.tracks[1].pianoroll.shape[0] == 192


def test_save_load(multitrack, tmp_path):
    filepath = str(tmp_path / "test.npz")
    multitrack.save(filepath)
    loaded = pypianoroll.load(filepath)
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
    filepath = str(tmp_path / "test.mid")
    multitrack.write(filepath)
    loaded = pypianoroll.read(filepath)
    assert loaded.name == multitrack.name
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


@fixture
def multitrack_to_merge():
    pianoroll_1 = np.zeros((192, 128), np.uint8)
    pianoroll_1[:191, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        program=0, is_drum=False, name="track_1", pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((192, 128), np.uint8)
    pianoroll_2[:191, [60, 64, 67, 72]] = 100
    track_2 = StandardTrack(
        program=0, is_drum=True, name="track_2", pianoroll=pianoroll_2
    )
    downbeat = np.zeros((192, 1), bool)
    downbeat[[0, 96]] = True
    return Multitrack(
        resolution=24, downbeat=downbeat, tracks=[track_1, track_2]
    )


def test_blend_any(multitrack_to_merge):
    blended = multitrack_to_merge.blend("any")
    assert blended.dtype == np.bool_
    assert not blended[0, 0]
    assert blended[0, 60]


def test_blend_sum(multitrack_to_merge):
    blended = multitrack_to_merge.blend("sum")
    assert blended.dtype == np.uint8
    assert blended[0, 0] == 0
    assert blended[0, 60] == 127


def test_blend_max(multitrack_to_merge):
    blended = multitrack_to_merge.blend("max")
    assert blended.dtype == np.uint8
    assert blended[0, 0] == 0
    assert blended[0, 60] == 100


def test_remove_empty():
    pianoroll_1 = np.zeros((192, 128), np.uint8)
    pianoroll_1[0:191, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        program=0, is_drum=False, name="track_1", pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), np.bool)
    track_2 = StandardTrack(
        program=0, is_drum=True, name="track_2", pianoroll=pianoroll_2
    )
    downbeat = np.zeros((192, 1), bool)
    downbeat[[0, 96]] = True
    multitrack = Multitrack(
        resolution=24, downbeat=downbeat, tracks=[track_1, track_2]
    )
    multitrack.remove_empty()
    assert len(multitrack) == 1

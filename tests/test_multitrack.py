"""Test cases for Multitrack class."""
import numpy as np
from pytest import fixture

import pypianoroll
from pypianoroll import Multitrack, Track


@fixture
def multitrack():
    pianoroll_1 = np.zeros((192, 128), np.uint8)
    pianoroll_1[:191, [60, 64, 67, 72]] = 100
    track_1 = Track(
        program=0, is_drum=False, name="track_1", pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((192, 128), np.bool)
    pianoroll_2[:191:16, 36] = True
    track_2 = Track(
        program=0, is_drum=True, name="track_2", pianoroll=pianoroll_2
    )
    return Multitrack(
        resolution=24, downbeat=[0, 96], tracks=[track_1, track_2],
    )


def test_slice(multitrack):
    sliced = multitrack[1, 20:40]
    assert isinstance(sliced, Multitrack)
    assert len(sliced.tracks) == 1
    assert sliced.tracks[0].pianoroll.shape == (20, 128)


def test_append_track(multitrack):
    pianoroll = np.zeros((192, 128), np.bool_)
    pianoroll[:191:16, 41] = True
    track_to_append = Track(name="track_3", pianoroll=pianoroll)
    multitrack.append(track_to_append)
    assert len(multitrack.tracks) == 3
    assert multitrack.tracks[2].name == "track_3"


def test_get_active_length(multitrack):
    assert multitrack.get_active_length() == 191


def test_get_active_pitch_range(multitrack):
    assert multitrack.get_active_pitch_range() == (36, 72)


def test_get_downbeat_steps(multitrack):
    assert multitrack.get_downbeat_steps() == [0, 96]


def test_get_max_length(multitrack):
    assert multitrack.get_max_length() == 192


def test_count_downbeat(multitrack):
    assert multitrack.count_downbeat() == 2


def test_get_stacked_pianoroll(multitrack):
    stacked = multitrack.get_stacked_pianoroll()
    assert stacked.shape == (192, 128, 2)


def test_remove_tracks(multitrack):
    multitrack.remove_tracks(1)
    assert len(multitrack.tracks) == 1


def test_trim_trailing_silence(multitrack):
    multitrack.trim_trailing_silence()
    assert multitrack.tracks[0].pianoroll.shape[0] == 191
    assert multitrack.tracks[1].pianoroll.shape[0] == 191


def test_pad_to_same(multitrack):
    pianoroll_1 = np.zeros((192, 128), np.uint8)
    pianoroll_1[0:191, [60, 64, 67, 72]] = 100
    track_1 = Track(
        program=0, is_drum=False, name="track_1", pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), np.bool)
    pianoroll_2[0:95:16, 36] = True
    track_2 = Track(
        program=0, is_drum=True, name="track_2", pianoroll=pianoroll_2
    )
    multitrack = Multitrack(
        resolution=24, downbeat=[0, 96], tracks=[track_1, track_2]
    )
    multitrack.pad_to_same()
    assert multitrack.tracks[0].pianoroll.shape[0] == 192
    assert multitrack.tracks[1].pianoroll.shape[0] == 192


def test_save_load(multitrack, tmp_path):
    """Test methods `Multitrack.save()` and `Multitrack.load()`."""
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
    """Test methods `Multitrack.write()` and `Multitrack.read()`."""
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
    track_1 = Track(
        program=0, is_drum=False, name="track_1", pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((192, 128), np.uint8)
    pianoroll_2[:191, [60, 64, 67, 72]] = 100
    track_2 = Track(
        program=0, is_drum=True, name="track_2", pianoroll=pianoroll_2
    )
    return Multitrack(
        resolution=24, downbeat=[0, 96], tracks=[track_1, track_2]
    )


def test_get_merged_pianoroll_any(multitrack_to_merge):
    merged = multitrack_to_merge.get_merged_pianoroll("any")
    assert np.issubdtype(merged.dtype, np.bool_)
    assert not merged[0, 0]
    assert merged[0, 60]


def test_get_merged_pianoroll_sum(multitrack_to_merge):
    merged = multitrack_to_merge.get_merged_pianoroll("sum")
    assert np.issubdtype(merged.dtype, np.integer)
    assert merged[0, 0] == 0
    assert merged[0, 60] == 200


def test_get_merged_pianoroll_max(multitrack_to_merge):
    merged = multitrack_to_merge.get_merged_pianoroll("max")
    assert np.issubdtype(merged.dtype, np.uint8)
    assert merged[0, 0] == 0
    assert merged[0, 60] == 100


def test_merge_tracks(multitrack_to_merge):
    multitrack_to_merge.merge_tracks([0, 1], "sum", remove_source=True)
    assert len(multitrack_to_merge.tracks) == 1


def test_get_empty_tracks():
    pianoroll_1 = np.zeros((192, 128), np.uint8)
    pianoroll_1[0:191, [60, 64, 67, 72]] = 100
    track_1 = Track(
        program=0, is_drum=False, name="track_1", pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), np.bool)
    track_2 = Track(
        program=0, is_drum=True, name="track_2", pianoroll=pianoroll_2
    )
    multitrack = Multitrack(
        resolution=24, downbeat=[0, 96], tracks=[track_1, track_2]
    )
    assert multitrack.get_empty_tracks() == [1]

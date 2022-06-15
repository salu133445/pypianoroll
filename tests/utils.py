"""Utility functions."""
import numpy as np
from pytest import fixture

from pypianoroll import BinaryTrack, Multitrack, StandardTrack


@fixture
def multitrack():
    pianoroll_1 = np.zeros((96, 128), np.uint8)
    pianoroll_1[:95, [60, 64, 67, 72]] = 100
    track_1 = StandardTrack(
        name="track_1", program=0, is_drum=False, pianoroll=pianoroll_1
    )
    pianoroll_2 = np.zeros((96, 128), bool)
    pianoroll_2[:95:16, 36] = True
    track_2 = BinaryTrack(
        name="track_2", program=0, is_drum=True, pianoroll=pianoroll_2
    )
    beat = np.zeros((96,), bool)
    beat[:96:24] = True
    downbeat = np.zeros((96,), bool)
    downbeat[0] = True
    return Multitrack(
        name="test",
        resolution=24,
        beat=beat,
        downbeat=downbeat,
        tracks=[track_1, track_2],
    )

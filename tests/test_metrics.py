"""Test cases for metrics."""
import numpy as np
from pytest import fixture

from pypianoroll.metrics import (
    empty_beat_rate,
    n_pitch_classes_used,
    n_pitches_used,
)


@fixture
def pianoroll():
    pianoroll = np.zeros((96, 128), np.uint8)
    pianoroll[:24, [60, 64, 67, 72]] = 100
    pianoroll[73:96, [72, 76, 79, 84]] = 80
    return pianoroll


def test_empty_beat_rate(pianoroll):
    """Test the empty_beat_rate metric."""
    assert empty_beat_rate(pianoroll, 24) == 0.5


def test_n_pitches_used(pianoroll):
    """Test the n_pitches_used metric."""
    assert n_pitches_used(pianoroll) == 7


def test_n_pitch_classes_used(pianoroll):
    """Test the n_pitches_used metric."""
    assert n_pitch_classes_used(pianoroll) == 3

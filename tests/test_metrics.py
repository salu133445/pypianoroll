"""Test cases for metrics."""
from math import isnan

import numpy as np
from pytest import fixture

from pypianoroll.metrics import (
    empty_beat_rate,
    n_pitch_classes_used,
    n_pitches_used,
    pitch_range,
    pitch_range_tuple,
)


@fixture
def pianoroll():
    pianoroll = np.zeros((96, 128), np.uint8)
    pianoroll[:24, [60, 64, 67, 72]] = 100
    pianoroll[73:96, [72, 76, 79, 84]] = 80
    return pianoroll


def test_empty_beat_rate(pianoroll):
    assert empty_beat_rate(pianoroll, 24) == 0.5


def test_n_pitches_used(pianoroll):
    assert n_pitches_used(pianoroll) == 7


def test_n_pitch_classes_used(pianoroll):
    assert n_pitch_classes_used(pianoroll) == 3


def test_pitch_range_tuple(pianoroll):
    assert pitch_range_tuple(pianoroll) == (60, 84)


def test_pitch_range(pianoroll):
    assert pitch_range(pianoroll) == 24


def test_pitch_range_empty():
    pianoroll = np.zeros((96, 128), np.uint8)
    assert isnan(pitch_range(pianoroll))

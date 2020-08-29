"""Test cases for `track.py` module."""
import unittest
import numpy as np
from pypianoroll import metrics


class MetricTestCase(unittest.TestCase):
    """Test case for metric module."""

    def setUp(self):
        self.pianoroll = np.zeros((96, 128), np.uint8)
        self.pianoroll[:24, [60, 64, 67, 72]] = 100
        self.pianoroll[73:96, [72, 76, 79, 84]] = 80
        self.beat_resolution = 24

    def tearDown(self):
        self.pianoroll = None
        self.beat_resolution = None

    def test_empty_beat_rate(self):
        """Test the empty_beat_rate metric."""
        empty_beat_rate = metrics.empty_beat_rate(
            self.pianoroll, self.beat_resolution
        )
        self.assertEqual(empty_beat_rate, 0.5)

    def test_n_pitches_used(self):
        """Test the n_pitches_used metric."""
        self.assertEqual(metrics.n_pitches_used(self.pianoroll), 7)

    def test_n_pitches_used(self):
        """Test the n_pitches_used metric."""
        self.assertEqual(metrics.n_pitches_used(self.pianoroll), 7)

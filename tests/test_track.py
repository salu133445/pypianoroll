"""Test cases for `track.py` module."""
import unittest
import numpy as np
from pypianoroll import Track


class TrackTestCase(unittest.TestCase):
    """Test case for class `Track`."""

    def setUp(self):
        pianoroll = np.zeros((96, 128), np.uint8)
        pianoroll[:95, [60, 64, 67, 72, 76, 79, 84]] = 100
        self.track = Track(
            program=0, is_drum=False, name="test", pianoroll=pianoroll
        )

    def tearDown(self):
        self.track = None

    def test_slice(self):
        """Test slicing method."""
        sliced = self.track[20:40]
        self.assertIsInstance(sliced, Track)
        self.assertEqual(sliced.pianoroll.shape, (20, 128))

    def test_assign_constant(self):
        """Test method `Track.slice()`."""
        self.track.assign_constant(50)
        self.assertEqual(self.track.pianoroll[0, 60], 50)

    def test_binarize(self):
        """Test method `Track.binarize()`."""
        self.track.binarize()
        self.assertTrue(np.issubdtype(self.track.pianoroll.dtype, np.bool_))
        self.assertTrue(self.track.pianoroll[0, 60])
        self.assertFalse(self.track.pianoroll[0, 0])

    def test_clip(self):
        """Test method `Track.clip()`."""
        self.track.clip(10, 50)
        self.assertEqual(self.track.pianoroll[0, 0], 10)
        self.assertEqual(self.track.pianoroll[0, 60], 50)

    def test_pad(self):
        """Test method `Track.pad()`."""
        self.track.pad(10)
        self.assertEqual(self.track.pianoroll.shape, (106, 128))

    def test_pad_to_multiple(self):
        """Test method `Track.pad_to_multiple()`."""
        self.track.pad_to_multiple(25)
        self.assertEqual(self.track.pianoroll.shape, (100, 128))

    def test_get_active_length(self):
        """Test method `Track.get_active_length()`."""
        self.assertEqual(self.track.get_active_length(), 95)

    def test_get_active_pitch_range(self):
        """Test method `Track.get_active_pitch_range()`."""
        self.assertEqual(self.track.get_active_pitch_range(), (60, 84))

    def test_is_binarized(self):
        """Test method `Track.is_binarized()`."""
        self.assertFalse(self.track.is_binarized())

    def test_transpose(self):
        """Test method `Track.transpose()`."""
        self.track.transpose(5)
        self.assertEqual(self.track.pianoroll[0, 65], 100)

    def trim_trailing_silence(self):
        """Test method `Track.trailing_silence()`."""
        self.track.trim_trailing_silence()
        self.assertEqual(self.track.pianoroll[0], 95)


class BinaryTrackTestCase(unittest.TestCase):
    """Test case for for class `Track` with a binary-valued piano-roll."""

    def setUp(self):
        pianoroll = np.zeros((96, 128), bool)
        pianoroll[:95, [60, 64, 67, 72]] = True
        self.track = Track(
            program=0, is_drum=False, name="test", pianoroll=pianoroll
        )

    def tearDown(self):
        self.track = None

    def test_assign_constant(self):
        """Test method `Track.assign_constant()`."""
        self.track.assign_constant(50.0)
        self.assertTrue(np.issubdtype(self.track.pianoroll.dtype, np.floating))
        self.assertEqual(self.track.pianoroll[0, 60], 50.0)

    def test_is_binarized(self):
        """Test method `Track.is_binarized()`."""
        self.assertTrue(self.track.is_binarized())


if __name__ == "__main__":
    unittest.main()

"""Test cases for `multitrack.py` module."""
import unittest
import os
import shutil
import tempfile
import numpy as np
from pypianoroll import Multitrack, Track

class MultitrackTestCase(unittest.TestCase):
    """Test case for class `Multitrack`."""
    def setUp(self):
        pianoroll_1 = np.zeros((192, 128), np.uint8)
        pianoroll_1[:191, [60, 64, 67, 72]] = 100
        track_1 = Track(pianoroll_1, 0, False, 'track_1')
        pianoroll_2 = np.zeros((192, 128), np.bool)
        pianoroll_2[:191:16, 36] = True
        track_2 = Track(pianoroll_2, 0, True, 'track_2')
        self.multitrack = Multitrack(tracks=[track_1, track_2],
                                     downbeat=[0, 96], beat_resolution=24)

    def tearDown(self):
        self.multitrack = None

    def test_slice(self):
        """Test slicing method."""
        sliced = self.multitrack[1, 20:40]
        self.assertIsInstance(sliced, Multitrack)
        self.assertEqual(len(sliced.tracks), 1)
        self.assertEqual(sliced.tracks[0].pianoroll.shape, (20, 128))

    def test_append_empty_track(self):
        """Test method `Multitrack.append_track()`."""
        self.multitrack.append_track(name='track_3')
        self.assertEqual(len(self.multitrack.tracks), 3)
        self.assertEqual(self.multitrack.tracks[2].name, 'track_3')

    def test_append_track(self):
        """Test method `Multitrack.append_track()`."""
        pianoroll = np.zeros((192, 128), np.bool_)
        pianoroll[:191:16, 41] = True
        track_to_append = Track(pianoroll, name='track_3')
        self.multitrack.append_track(track_to_append)
        self.assertEqual(len(self.multitrack.tracks), 3)
        self.assertEqual(self.multitrack.tracks[2].name, 'track_3')

    def test_get_active_length(self):
        """Test method `Multitrack.get_active_length()`."""
        self.assertEqual(self.multitrack.get_active_length(), 191)

    def test_get_active_pitch_range(self):
        """Test method `Multitrack.get_active_pitch_range()`."""
        self.assertEqual(self.multitrack.get_active_pitch_range(), (36, 72))

    def test_get_downbeat_steps(self):
        """Test method `Multitrack.get_downbeat_steps()`."""
        self.assertEqual(self.multitrack.get_downbeat_steps(), [0, 96])

    def test_get_max_length(self):
        """Test method `Multitrack.get_max_length()`."""
        self.assertEqual(self.multitrack.get_max_length(), 192)

    def test_count_downbeat(self):
        """Test method `Multitrack.count_downbeat()`."""
        self.assertEqual(self.multitrack.count_downbeat(), 2)

    def test_get_stacked_pianoroll(self):
        """Test method `Multitrack.get_stacked_pianoroll()`."""
        stacked = self.multitrack.get_stacked_pianoroll()
        self.assertEqual(stacked.shape, (192, 128, 2))

    def test_remove_tracks(self):
        """Test method `Multitrack.remove_tracks()`."""
        self.multitrack.remove_tracks(1)
        self.assertEqual(len(self.multitrack.tracks), 1)

    def test_trim_trailing_silence(self):
        """Test method `Multitrack.trim_trailing_silence()`."""
        self.multitrack.trim_trailing_silence()
        self.assertEqual(self.multitrack.tracks[0].pianoroll.shape[0], 191)
        self.assertEqual(self.multitrack.tracks[1].pianoroll.shape[0], 191)

class MultitrackPadTestCase(unittest.TestCase):
    """Test case for pad methods for class `Multitrack`."""
    def test_pad_to_same(self):
        """Test method `Multitrack.pad_to_same()`."""
        pianoroll_1 = np.zeros((192, 128), np.uint8)
        pianoroll_1[0:191, [60, 64, 67, 72]] = 100
        track_1 = Track(pianoroll_1, 0, False, 'track_1')
        pianoroll_2 = np.zeros((96, 128), np.bool)
        pianoroll_2[0:95:16, 36] = True
        track_2 = Track(pianoroll_2, 0, True, 'track_2')
        multitrack = Multitrack(tracks=[track_1, track_2],
                                downbeat=[0, 96], beat_resolution=24)
        multitrack.pad_to_same()
        self.assertEqual(multitrack.tracks[0].pianoroll.shape[0], 192)
        self.assertEqual(multitrack.tracks[1].pianoroll.shape[0], 192)

class MultitrackMergeTestCase(unittest.TestCase):
    """Test case for merge methods for class `Multitrack`."""
    def setUp(self):
        pianoroll_1 = np.zeros((192, 128), np.uint8)
        pianoroll_1[:191, [60, 64, 67, 72]] = 100
        track_1 = Track(pianoroll_1, 0, False, 'track_1')
        pianoroll_2 = np.zeros((192, 128), np.uint8)
        pianoroll_2[:191, [60, 64, 67, 72]] = 100
        track_2 = Track(pianoroll_2, 0, True, 'track_2')
        self.multitrack = Multitrack(tracks=[track_1, track_2],
                                     downbeat=[0, 96], beat_resolution=24)

    def tearDown(self):
        self.multitrack = None

    def test_get_merged_pianoroll_any(self):
        """Test method `Multitrack.get_merged_pianoroll()`."""
        merged = self.multitrack.get_merged_pianoroll('any')
        self.assertTrue(np.issubdtype(merged.dtype, np.bool_))
        self.assertEqual(merged[0, 0], False)
        self.assertEqual(merged[0, 60], True)

    def test_get_merged_pianoroll_sum(self):
        """Test method `Multitrack.get_merged_pianoroll()`."""
        merged = self.multitrack.get_merged_pianoroll('sum')
        self.assertTrue(np.issubdtype(merged.dtype, np.integer))
        self.assertEqual(merged[0, 0], 0)
        self.assertEqual(merged[0, 60], 200)

    def test_get_merged_pianoroll_max(self):
        """Test method `Multitrack.get_merged_pianoroll()`."""
        merged = self.multitrack.get_merged_pianoroll('max')
        self.assertTrue(np.issubdtype(merged.dtype, np.uint8))
        self.assertEqual(merged[0, 0], 0)
        self.assertEqual(merged[0, 60], 100)

    def test_merge_tracks(self):
        """Test method `Multitrack.test_merge_tracks()`."""
        self.multitrack.merge_tracks([0, 1], 'sum', remove_merged=True)
        self.assertEqual(len(self.multitrack.tracks), 1)

class MultitrackEmptyTrackTestCase(unittest.TestCase):
    """Test case for class `Multitrack` with an empty track."""
    def test_get_empty_tracks(self):
        """Test method `Multitrack.get_empty_tracks()`."""
        pianoroll_1 = np.zeros((192, 128), np.uint8)
        pianoroll_1[0:191, [60, 64, 67, 72]] = 100
        track_1 = Track(pianoroll_1, 0, False, 'track_1')
        pianoroll_2 = np.zeros((96, 128), np.bool)
        track_2 = Track(pianoroll_2, 0, True, 'track_2')
        multitrack = Multitrack(tracks=[track_1, track_2],
                                downbeat=[0, 96], beat_resolution=24)
        self.assertEqual(multitrack.get_empty_tracks(), [1])

class MultitrackIOTestCase(unittest.TestCase):
    """Test case for IO methods for class `Multitrack`."""
    def setUp(self):
        pianoroll_1 = np.zeros((192, 128), np.uint8)
        pianoroll_1[:191, [60, 64, 67, 72]] = 100
        track_1 = Track(pianoroll_1, 0, False, 'track_1')
        pianoroll_2 = np.zeros((192, 128), np.bool)
        pianoroll_2[:191:16, 36] = True
        track_2 = Track(pianoroll_2, 0, True, 'track_2')
        self.multitrack = Multitrack(tracks=[track_1, track_2],
                                     downbeat=[0, 96], beat_resolution=24)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        self.multitrack = None
        shutil.rmtree(self.test_dir)

    def test_save_load(self):
        """Test methods `Multitrack.save()` and `Multitrack.load()`."""
        filepath = os.path.join(self.test_dir, 'test.npz')
        self.multitrack.save(filepath)
        loaded = Multitrack(filepath)
        self.assertTrue(np.allclose(loaded.tempo, self.multitrack.tempo))
        self.assertTrue(np.allclose(loaded.downbeat, self.multitrack.downbeat))
        self.assertEqual(loaded.beat_resolution,
                         self.multitrack.beat_resolution)
        self.assertEqual(loaded.name, self.multitrack.name)
        self.assertTrue(np.allclose(loaded.tracks[0].pianoroll,
                                    self.multitrack.tracks[0].pianoroll))
        self.assertEqual(loaded.tracks[0].program, 0)
        self.assertEqual(loaded.tracks[0].is_drum, False)
        self.assertEqual(loaded.tracks[0].name, 'track_1')
        self.assertTrue(np.allclose(loaded.tracks[1].pianoroll,
                                    self.multitrack.tracks[1].pianoroll))
        self.assertEqual(loaded.tracks[1].program, 0)
        self.assertEqual(loaded.tracks[1].is_drum, True)
        self.assertEqual(loaded.tracks[1].name, 'track_2')

    def test_write_parse(self):
        """Test methods `Multitrack.write()` and `Multitrack.parse_midi()`."""
        filepath = os.path.join(self.test_dir, 'test.mid')
        self.multitrack.write(filepath)
        loaded = Multitrack(filepath)
        self.assertTrue(np.allclose(loaded.tempo, self.multitrack.tempo))
        self.assertEqual(loaded.name, self.multitrack.name)
        self.assertTrue(np.allclose(loaded.tracks[0].pianoroll,
                                    self.multitrack.tracks[0].pianoroll))
        self.assertEqual(loaded.tracks[0].program, 0)
        self.assertEqual(loaded.tracks[0].is_drum, False)
        self.assertEqual(loaded.tracks[0].name, 'track_1')
        self.assertTrue(np.allclose((loaded.tracks[1].pianoroll > 0),
                                    self.multitrack.tracks[1].pianoroll))
        self.assertEqual(loaded.tracks[1].program, 0)
        self.assertEqual(loaded.tracks[1].is_drum, True)
        self.assertEqual(loaded.tracks[1].name, 'track_2')

if __name__ == '__main__':
    unittest.main()

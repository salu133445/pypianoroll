"""
Class for multi-track piano-rolls with metadata.
"""
import warnings
from copy import deepcopy
import numpy as np
import pretty_midi
from scipy.sparse import csc_matrix
from .track import Track

class MultiTrack(object):
    """
    A multi-track piano-roll container

    Attributes
    ----------
    tracks : list
        List of :class:`pianoroll.Track` objects.
    tempo : np.ndarray, shape=(num_time_step,), dtype=float
        Tempo array that indicates the tempo value (in bpm) at each time
        step. Length is the number of time steps.
    downbeat : np.ndarray, shape=(num_time_step,), dtype=bool
        Downbeat array that indicates whether the time step contains a
        downbeat, i.e. the first time step of a bar. Length is the number of
        time steps.
    beat_resolution : int
        Resolution of a beat (in time step).
    name : str
        Name of the multi-track piano-roll.

    Notes
    -----
    The length of ``tempo`` and ``downbeat`` can be different. During
    conversion and rendering, the tempo array will be automatically padded
    to appropriate length with the last tempo value. The downbeat array has
    no effect on playback.
    """
    def __init__(self, filepath=None, tracks=None, tempo=120.0, downbeat=None,
                 beat_resolution=24, name='unknown'):
        """
        Initialize by parsing MIDI file or loading .npz file or creating minimal
        empty instance.

        Parameters
        ----------
        filepath : str
            File path to a MIDI file (.mid, .midi, .MID, .MIDI) or a .npz file.
        tracks : list
            List of :class:`pianoroll.Track` objects to be added to the track
            list when ``filepath`` is not provided.
        tempo : int or np.ndarray, shape=(num_time_step,), dtype=float
            Tempo array that indicates the tempo value (in bpm) at each time
            step. Length is the number of time steps. Will be assigned to
            ``tempo`` when ``filepath`` is not provided. If an integer is
            provided, it will be first converted to a numpy array. Default to
            120.0.
        downbeat : np.ndarray, shape=(num_time_step,), dtype=bool
            Downbeat array that indicates whether the time step contains a
            downbeat, i.e. the first time step of a bar. Length is the number of
            time steps. Will be assigned to ``downbeat`` when ``filepath`` is
            not provided.
        beat_resolution : int
            Resolution of a beat (in time step). Will be assigned to
            ``beat_resolution`` when ``filepath`` is not provided. Default to
            24.
        name : str
            Name to be assigned to the multi-track piano-roll. Default to
            'unknown'.

        Notes
        -----
        When ``filepath`` is given, ignore arguments ``tracks``, ``tempo``,
        ``downbeat`` and ``name``.
        """
        # parse input file
        if filepath is not None:
            if not isinstance(filepath, str):
                raise TypeError("`filepath` must be of str type")
            if filepath.endswith(('.mid', '.midi', '.MID', '.MIDI')):
                self.parse_midi(filepath)
                self.check_validity()
                self.name = name
                warnings.warn("ignore arguments `tracks`, `tempo` and "
                              "`downbeat`", RuntimeWarning)
            elif filepath.endswith('.npz'):
                self.load(filepath)
                self.check_validity()
                warnings.warn("ignore arguments `tracks`, `tempo`, `downbeat` "
                              "and `name`", RuntimeWarning)
            else:
                raise ValueError("Unsupported file type")
        else:
            if tracks is not None:
                self.tracks = tracks
            else:
                self.tracks = [Track()]
            if isinstance(tempo, float):
                self.tempo = np.array(tempo, float)
            else:
                self.tempo = tempo
            self.downbeat = downbeat
            self.beat_resolution = beat_resolution
            self.name = name
            self.check_validity()

    def append_track(self, track=None, pianoroll=None, program=0, is_drum=False,
                     lowest_pitch=0, name='unknown'):
        """
        Append a multitrack.Track instance to the track list or create a new
        multitrack.Track object and append it to the track list.

        Parameters
        ---------
        track : pianoroll.Track
            A :class:`pianoroll.Track` instance to be appended to the track
            list.
        pianoroll : np.ndarray, shape=(num_time_step, num_pitch)
            Piano-roll matrix. First dimension represents time. Second dimension
            represents pitch. The lowest pitch is given by ``lowest_pitch``.
        program: int
            Program number according to General MIDI specification. Available
            values are 0 to 127. Default to 0 (Acoustic Grand Piano).
        is_drum : bool
            Drum indicator. True for drums. False for other instruments. Default
            to False.
        name : str
            Name of the track. Default to 'unknown'.
        lowest_pitch : int
            Indicate the lowest pitch in the piano-roll. Available values are 0
            to 127.
        """
        if track is not None:
            if not isinstance(track, Track):
                raise TypeError("`track` must be a multitrack.Track instance")
            track.check_validity()
        else:
            track = Track(pianoroll=pianoroll, program=program, is_drum=is_drum,
                          lowest_pitch=lowest_pitch, name=name)
        self.tracks.append(track)

    def binarize(self, threshold=0.0):
        """
        Binarize the piano-rolls of all tracks. Pass the track if its piano-roll
        is already binarized

        Parameters
        ----------
        threshold : float
            Threshold to binarize the piano-rolls. Default to zero.
        """
        for track in self.tracks:
            track.binarize(threshold)

    def check_validity(self):
        """"Raise error if any invalid attribute found"""
        # tracks
        for idx, track in enumerate(self.tracks):
            if not isinstance(track, Track):
                raise TypeError("`tracks` must be multitrack.Track instances")
            try:
                track.check_validity()
            except:
                print("`tracks[{}]` is not a valid multitrack.Track instance"
                      .format(idx))
                raise
        # tempo
        if not isinstance(self.tempo, np.ndarray):
            raise TypeError("`tempo` must be of np.ndarray type")
        if not (np.issubdtype(self.tempo.dtype, np.int),
                np.issubdtype(self.tempo.dtype, np.float)):
            raise TypeError("Data type of `tempo` must be int or float.")
        if self.tempo.ndim > 2:
            raise ValueError("`tempo` must be a 1D numpy array")
        if np.any(self.tempo <= 0.0):
            raise ValueError("`tempo` must contains only positive numbers")
        # downbeat
        if self.downbeat is not None:
            if not isinstance(self.downbeat, np.ndarray):
                raise TypeError("`downbeat` must be of np.ndarray type")
            if not np.issubdtype(self.downbeat.dtype, np.bool):
                raise TypeError("Data type of `downbeat` must be bool.")
            if self.downbeat.ndim > 1:
                raise ValueError("`downbeat` must be a 1D numpy array")
        # beat_resolution
        if not isinstance(self.beat_resolution, int):
            raise TypeError("`beat_resolution` must be of int type")
        if self.beat_resolution < 1:
            raise ValueError("`beat_resolution` must be a positive integer")
        # name
        if not isinstance(self.name, str):
            raise TypeError("`name` must be of str type")

    def compress_pitch_range(self):
        """Compress the piano-rolls of all tracks to active pitch range"""
        for track in self.tracks:
            track.compress_pitch_range()

    def copy(self):
        """
        Return a copy of the object

        Returns
        -------
        copied : `pianoroll.MultiTrack` object
            A copy of the object.
        """
        copied = deepcopy(self)
        return copied

    def get_downbeat_steps(self):
        """
        Return the indices of time steps that contain downbeats

        Returns
        -------
        downbeat_steps : list
            List of indices of time steps that contain downbeats.
        """
        downbeat_steps = np.nonzero(self.downbeat)[0].astype(int).tolist()
        return downbeat_steps

    def get_length(self, track_indices=None):
        """
        Return length (in time step) of tracks specified by ``track_indices``.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to be collected. If None
            (by default), all tracks will be collected.
        """
        if track_indices is None:
            track_indices = range(len(self.tracks))
        return max([self.tracks[idx].get_length() for idx in track_indices] +
                   [int(self.downbeat.shape[0]), int(self.tempo.shape[0])])

    def get_merged_pianoroll(self, track_indices=None, binarized=True,
                             threshold=0.0, use_bool_sum=True):
        """
        Return a merged piano-roll of tracks specified by ``track_indices``.

        Notes
        -----
        If ``binarized`` is True, the (copied) collected piano-rolls will be
        binarized first. When ``binarized`` is True or the collected piano-rolls are
        already binarized, boolean summation is performed if
        ``use_bool_sum`` is True. Otherwise, integer summation is
        performed.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to be collected. If None
            (by default), all tracks will be collected.
        binarized : bool
            If True, boolean summation on binarized piano-rolls is preformed.
            Otherwise, perform integer summation on raw piano-rolls. Default to
            True.
        threshold : int
            Threshold to binarize the collected piano-rolls. Only effective when
            ``binarized`` is True. Default to zero.
        use_bool_sum : bool
            If true, boolean summation is performed. Otherwise, integer
            summation is performed. Only effective when the piano-rolls are
            already binarized or ``binarized`` is True. Default to True.

        Retruns
        -------
        merged : np.ndarray
            The merged piano-rolls. Ordering follows what appear in
            ``track_indices``. Size of the numpy arrays are (num_time_step,
            num_pitch).
        lowest : int
            Indicate the lowest pitch in the merged piano-roll.
        """
        stacked, lowest = self.get_stacked_pianorolls(track_indices, binarized,
                                                      threshold)

        if binarized and use_bool_sum:
            merged = np.sum(stacked, axis=3, dtype=bool)
        else:
            merged = np.sum(stacked, axis=3)

        return merged, lowest

    def get_num_bar(self):
        """
        Return the number of bars. The return value is calculated based solely
        on the downbeat array

        Returns
        -------
        num_bar : int
            The number of bars according to the downbeat array
        """
        num_bar = np.sum(self.downbeat, dtype=int)
        return num_bar

    def get_num_track(self):
        """
        Return the number of tracks

        Returns
        -------
        num_track : int
            The number of tracks.
        """
        num_track = len(self.tracks)
        return num_track

    def get_pitch_range(self, track_indices=None):
        """
        Return the pitch range in tuple (lowest, highest) of the piano-rolls
        specified by ``track_indices``.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to be collected. If None
            (by default), all tracks will be collected.

        Returns
        -------
        lowest : int
            Indicate the lowest pitch in the collected piano-rolls.
        highest : int
            Indicate the highest pitch in the collected piano-rolls.
        """
        if track_indices is None:
            track_indices = range(len(self.tracks))

        lowest, highest = self.tracks[track_indices[0]].get_pitch_range()
        for idx in track_indices[1:]:
            low, high = self.tracks[idx].get_pitch_range()
            if low < lowest:
                lowest = low
            if high > highest:
                highest = high

        return lowest, highest

    def get_stacked_pianorolls(self, track_indices=None, binarized=False,
                               threshold=0.0):
        """
        Return a stacked multi-track piano-roll composed of tracks specified
        by ``track_indices``. The shape of the return np.ndarray is
        (num_time_step, num_pitch, num_track).

        Parameters
        ----------
        track_indices : list
            List of indices that indicate which tracks to be collected. If None
            (by default), all tracks will be collected.
        binarized : bool
            If True, stack binarized copies of the collected piano-rolls.
            Otherwise, stack the original piano-rolls. Default to False.
        threshold : int
            Threshold to binarize the collected piano-rolls. Only effective when
            ``binarized`` is True. Default to zero.

        Returns
        -------
        stacked : np.ndarray, shape=(num_time_step, num_pitch, num_track)
            The stacked piano-roll. Shape is (num_time_step, num_pitch,
            num_track). The ordering of tracks follows what appear in
            ``track_indices``.
        lowest : int
            Indicate the lowest pitch in the stacked piano-roll.
        """
        if track_indices is None:
            track_indices = range(len(self.tracks))

        lowest, highest = self.get_pitch_range(track_indices)
        length = self.get_length()

        to_stack = []
        for idx in track_indices:
            to_pad_l = self.tracks[idx].lowest_pitch - lowest
            to_pad_h = (highest - self.tracks[idx].lowest_pitch
                        - self.tracks[idx].pianoroll.shape[1])
            to_pad_t = length - self.tracks[idx].pianoroll.shape[0]
            to_pad = ((0, to_pad_t), (to_pad_l, to_pad_h))
            if binarized:
                binarized = (self.tracks[idx].pianoroll > threshold)
                padded = np.lib.pad(binarized, to_pad, 'constant',
                                    constant_values=((0, 0), (0, 0)))
            else:
                padded = np.lib.pad(self.tracks[idx].pianoroll, to_pad,
                                    'constant',
                                    constant_values=((0, 0), (0, 0)))
            to_stack.append(padded)

        stacked = np.stack(to_stack, -1)
        return stacked, lowest

    def is_binarized(self, track_indices=None):
        """
        Return True if pianorolls specified by ``track_indices`` are already
        binarized. Otherwise, return False

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to be collected. If None
            (by default), all tracks will be collected.

        Returns
        -------
        is_binarized : bool
            True if all the collected piano-rolls are already binarized;
            otherwise, False.
        """
        if track_indices is None:
            track_indices = range(len(self.tracks))
        for idx in track_indices:
            if not self.tracks[idx].is_binarized():
                return False
        return True

    def load(self, filepath):
        pass
        # """Save a .npz file to a multi-track piano-roll"""
        # def reconstruct_sparse_matrix(target_dict, name):
        #     """
        #     Return the reconstructed scipy.sparse.csc_matrix, whose components
        #     are stored in ``target_dict`` with prefix given as ``name``
        #     """
        #     return csc_matrix((target_dict[name+'_csc_data'],
        #                        target_dict[name+'_csc_indices'],
        #                        target_dict[name+'_csc_indptr']),
        #                       shape=target_dict[name+'_csc_shape'])
        # # load the .npz file
        # with np.load(filepath) as loaded:
        #     pianoroll_component_count = 0
        #     for filename in loaded.files:
        #         if filename.startswith('pianorolls_'):
        #             pianoroll_component_count += 1
        #     # sort the file names in o
        #     for idx in range(len(pianoroll_files)/4):
        #         # reconstruct csc_matrix and add it to csc_matrix dictionary
        #         self.pianorolls[idx] = reconstruct_sparse_matrix(loaded, \
        #             'pianorolls_{:03d}'.format(idx))
        #     self.downbeat = loaded['downbeat']
        # self.check_validity()

    def merge_tracks(self, track_indices=None, program=0, is_drum=False,
                     name='merged', remove_merged=False, binarized=True,
                     threshold=0.0, use_bool_sum=True):
        """
        Merge piano-rolls of tracks specified by ``track_indices``. The merged
        track will have program number as given by ``program`` and drum
        indicator as given by ``is_drum``. The merged track will be appended at
        the end of the track list.

        Note
        ----
        If ``binarized`` is True, the piano-rolls will be binarized first. When
        ``binarized`` is True or the piano-rolls are already binarized, boolean
        summation is performed if ``use_bool_sum`` is True. Otherwise,
        integer summation is performed.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to collect. If None (by
            default), all tracks will be collectted.
        program: int
            Program number to be assigned to the merged track. Available values
            are 0 to 127.
        is_drum : bool
            Drum indicator to be assigned to the merged track.
        name : str
            Name to be assigned to the merged track. Default to 'merged'.
        remove_merged : bool
            True to remove the merged tracks from the track list. False to keep
            them. Default to False.
        binarized : bool
            True to merge binarized copies of the collected piano-rolls. False
            to merge the original piano-rolls. Default to True.
        threshold : int
            Threshold to binarize the collected piano-rolls. Only effective when
            ``make_binary`` is True. Default to zero.
        use_bool_sum : bool
            If true, boolean summation is performed. Otherwise, integer
            summation is performed. Only effective when the piano-rolls are
            already binarized or ``binarized`` is True. Default to True.
        """
        merged, lowest = self.get_merged_pianoroll(track_indices, binarized,
                                                   threshold, use_bool_sum)

        merged_track = Track(merged, program, is_drum, lowest, name)
        self.append_track(merged_track)

        if remove_merged:
            self.remove_tracks(track_indices)

    def parse_midi(self, filepath):
        """
        Parse a MIDI file

        Parameters
        ----------
        filepath : str
            The path to the MIDI file.
        """
        pass
        # """Parse the MIDI file given by ``filepath``"""
        # pm = pretty_midi.PrettyMIDI(filepath)
        # self.pianorolls = get_pianorolls(pm, None, \
        #     self.beat_resolution, self.binarized, True)
        # self.name = os.path.basename(filepath)

    def remove_tracks(self, track_indices):
        """
        Remove tracks specified by ``track_indices``.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to remove.
        """
        self.tracks = [track for idx, track in enumerate(self.tracks)
                       if idx not in track_indices]

    def save(self, filepath, compressed=True):
        """
        Save to a .npz file

        Parameters
        ----------
        filepath : str
            The path to write the .npz file.
        """
        pass
        # """Save the multi-track piano-roll to a .npz file"""
        # def update_sparse_matrix(target_dict, sparse_matrix, name):
        #     """
        #     Turn ``sparse_matrix`` into a scipy.sparse.csc_matrix and update its
        #     component arrays to the ``target_dict`` with key given as ``name``
        #     """
        #     csc = csc_matrix(sparse_matrix)
        #     target_dict[name+'_csc_data'] = csc.data
        #     target_dict[name+'_csc_indices'] = csc.indices
        #     target_dict[name+'_csc_indptr'] = csc.indptr
        #     target_dict[name+'_csc_shape'] = csc.shape
        # result_dict = {'downbeat': self.downbeat}
        # for idx, pianoroll in enumerate(self.pianorolls):
        #     update_sparse_matrix(result_dict, pianoroll,
        #                          'pianorolls_{:03d}'.format(idx))

        # if not filepath.endswith('.npz'):
        #     filepath += '.npz'

        # if compressed:
        #     np.savez_compressed(filepath, **result_dict)
        # else:
        #     np.savez(filepath, **result_dict)

    def to_pretty_midi(self):
        """
        Convert to a :class:`pretty_midi.PrettyMIDI` instance

        Returns
        -------
        pm : `pretty_midi.PrettyMIDI` object
            The converted :class:`pretty_midi.PrettyMIDI` instance.
        """
        pass

        # # create a PrettyMIDI class instance
        # if self.tempo:
        #     pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo[0])
        # else:
        #     pm = pretty_midi.PrettyMIDI()
        # # iterate through all the input instruments
        # for idx, pianoroll in enumerate(self.pianorolls):
        #     instrument = get_instrument(pianoroll, self.program[idx],
        #                                 self.is_drum[idx], tempo,
        #                                 self.beat_resolution)
        #     pm.instruments.append(instrument)
        # return pm

    def transpose(self, semitone):
        """
        Transpose the piano-rolls by ``semitones`` semitones

        Parameters
        ----------
        semitone : int
            Number of semitones transpose the piano-rolls.
        """
        for track in self.tracks():
            track.transpose(semitone)

    def trim_trailing_silence(self):
        """Trim the trailing silence of the piano-roll"""
        for track in self.tracks():
            track.trim_trailing_silence()

    def write_midi(self, filepath):
        """
        Write to a MIDI file

        Parameters
        ----------
        filepath : str
            The path to write the MIDI file.
        """
        pm = self.to_pretty_midi()
        pm.write(filepath)

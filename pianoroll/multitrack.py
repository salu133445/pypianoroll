"""
Class for multi-track piano-rolls with metadata.
"""
from __future__ import division
import warnings
import json
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
    The length of `tempo` and `downbeat` can be different. During conversion
    and rendering, the tempo array will be automatically padded to
    appropriate length with the last tempo value. The downbeat array has no
    effect on playback.
    """
    def __init__(self, filepath=None, filepath_json=None, tracks=None,
                 tempo=120.0, downbeat=None, beat_resolution=24,
                 name='unknown'):
        """
        Initialize by one of the following ways
        - parsing a MIDI file
        - loading a .npz file and a JSON file
        - assigning values for attributes

        Notes
        -----
        When `filepath` is given, ignore arguments `tracks`, `tempo`, `downbeat`
        and `name`.

        Parameters
        ----------
        filepath : str
            File path to a MIDI file (.mid, .midi, .MID, .MIDI) to be parsed or
            a .npz file to be loaded.
        filepath_json : str
            File path to a JSON file to be loaded. Required and only effective
            when `filepath` is a .npz file.
        beat_resolution : int
            Resolution of a beat (in time step). Will be assigned to
            `beat_resolution` when `filepath` is not provided. Default to 24.
        tracks : list
            List of :class:`pianoroll.Track` objects to be added to the track
            list when `filepath` is not provided.
        tempo : int or np.ndarray, shape=(num_time_step,), dtype=float
            Tempo array that indicates the tempo value (in bpm) at each time
            step. Length is the number of time steps. Will be assigned to
            `tempo` when `filepath` is not provided. If an integer is provided,
            it will be first converted to a numpy array. Default to 120.0.
        downbeat : np.ndarray, shape=(num_time_step,), dtype=bool
            Downbeat array that indicates whether the time step contains a
            downbeat, i.e. the first time step of a bar. Length is the number of
            time steps. Will be assigned to `downbeat` when `filepath` is not
            provided.
        name : str
            Name to be assigned to the multi-track piano-roll. Default to
            'unknown'.
        """
        # parse input file
        if filepath is not None:
            if not isinstance(filepath, str):
                raise TypeError("`filepath` must be of str type")
            if filepath.endswith(('.mid', '.midi', '.MID', '.MIDI')):
                self.beat_resolution = beat_resolution
                self.parse_midi(filepath)
                self.name = name
                warnings.warn("ignore arguments `tracks`, `tempo` and "
                              "`downbeat`", RuntimeWarning)
            elif filepath.endswith('.npz'):
                if filepath_json is None:
                    raise TypeError("`filepath_json` must be given when "
                                    "`filepath` is a .npz file")
                self.load(filepath, filepath_json)
                warnings.warn("ignore arguments `tracks`, `tempo`, `downbeat` "
                              "and `name`", RuntimeWarning)
            else:
                raise ValueError("Unsupported file type")
        else:
            if tracks is not None:
                self.tracks = tracks
            else:
                self.tracks = [Track()]
            if isinstance(self.tempo, (int, float)):
                self.tempo = np.array([self.tempo])
            else:
                self.tempo = tempo
            self.downbeat = downbeat
            self.beat_resolution = beat_resolution
            self.name = name
            self.check_validity()

    def append_track(self, track=None, pianoroll=None, program=0, is_drum=False,
                     lowest=0, name='unknown'):
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
            represents pitch. The lowest pitch is given by `lowest`.
            Available datatypes are bool, int, float.
        program: int
            Program number according to General MIDI specification [1].
            Available values are 0 to 127. Default to 0 (Acoustic Grand Piano).
        is_drum : bool
            Drum indicator. True for drums. False for other instruments. Default
            to False.
        name : str
            Name of the track. Default to 'unknown'.
        lowest : int
            Indicate the lowest pitch of the piano-roll. Default to zero.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set
        """
        if track is not None:
            if not isinstance(track, Track):
                raise TypeError("`track` must be a multitrack.Track instance")
            track.check_validity()
        else:
            track = Track(pianoroll, program, is_drum, name, lowest)
        self.tracks.append(track)

    def binarize(self, threshold=0):
        """
        Binarize the piano-rolls of all tracks. Pass the track if its piano-roll
        is already binarized

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano-rolls. Default to zero.
        """
        for track in self.tracks:
            track.binarize(threshold)

    def check_validity(self):
        """"
        Raise error if any invalid attribute found

        Raises
        ------
        TypeError
            If contain any attribute with invalid type.
        ValueError
            If contain any attribute with invalid value (of the correct type).
        """
        # tracks
        for track in self.tracks:
            if not isinstance(track, Track):
                raise TypeError("`tracks` must be `multitrack.Track` instances")
            track.check_validity()
        # tempo
        if not isinstance(self.tempo, np.ndarray):
            raise TypeError("`tempo` must be of int or np.ndarray type")
        elif not (np.issubdtype(self.tempo.dtype, np.int),
                  np.issubdtype(self.tempo.dtype, np.float)):
            raise TypeError("Data type of `tempo` must be int or float.")
        elif self.tempo.ndim != 1:
            raise ValueError("`tempo` must be a 1D numpy array")
        if np.any(self.tempo <= 0.0):
            raise ValueError("`tempo` must contains only positive numbers")
        # downbeat
        if self.downbeat is not None:
            if not isinstance(self.downbeat, np.ndarray):
                raise TypeError("`downbeat` must be of np.ndarray type")
            if not np.issubdtype(self.downbeat.dtype, np.bool):
                raise TypeError("Data type of `downbeat` must be bool.")
            if self.downbeat.ndim != 1:
                raise ValueError("`downbeat` must be a 1D numpy array")
        # beat_resolution
        if not isinstance(self.beat_resolution, int):
            raise TypeError("`beat_resolution` must be of int type")
        if self.beat_resolution < 1:
            raise ValueError("`beat_resolution` must be a positive integer")
        # name
        if not isinstance(self.name, str):
            raise TypeError("`name` must be of str type")

    def clip(self, lower=0, upper=128):
        """
        Clip the piano-rolls of all tracks by an lower bound and an upper bound
        specified by `lower` and `upper`, respectively

        Parameters
        ----------
        lower : int or float
            The lower bound to clip the piano-roll. Default to 0.
        upper : int or float
            The upper bound to clip the piano-roll. Default to 128.
        """
        for track in self.tracks:
            track.clip(lower, upper)

    def compress(self):
        """Compress the piano-rolls of all tracks to active pitch range"""
        for track in self.tracks:
            track.compress()

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

    def expand(self, lowest=0, highest=127):
        """
        Expand the piano-rolls of all tracks to a pitch range specified by
        `lowest` and `highest`
        """
        for track in self.tracks:
            track.expand(lowest, highest)

    def get_downbeat_steps(self):
        """
        Return the indices of time steps that contain downbeats

        Returns
        -------
        downbeat_steps : np.ndarray
            Indices of time steps that contain downbeats.
        """
        downbeat_steps = np.nonzero(self.downbeat)[0]
        return downbeat_steps

    def get_length(self, track_indices=None):
        """
        Return length (in time step) of tracks specified by `track_indices`.

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

    def get_merged_pianoroll(self, track_indices=None, mode='sum',
                             clipped=True, upper=128):
        """
        Return a merged piano-roll of tracks specified by `track_indices`.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to be collected. If None
            (by default), all tracks will be collected.
        mode : {'sum', 'max', 'any'}
            Indicate the merging function to apply along the track axis. Default
            to 'sum'.
            - In 'sum' mode, the piano-roll of the merged track is the summation
              of the collected piano-rolls. Note that for binarized piano-roll,
              integer summation is performed.
            - In 'max' mode, for each pixel, the maximal value among the
              collected piano-rolls is assigned to the merged piano-roll.
            - In 'any' mode, the value of a pixel in the merged piano-roll is
              True if any of the collected piano-rolls has nonzero value at that
              pixel; False if all piano-rolls are inactive (zero-valued) at that
              pixel.
        clipped : bool
            True to clip the velocities of the parsed piano-rolls by `upper`.
            False to use the raw sum of the velocities. Only effective in 'sum'
            mode. Default to True.
        upper : int
            The upper bound to clip the input piano-roll. Only effective when
            'clipped' is True. Default to 128.

        Retruns
        -------
        merged : np.ndarray, shape=(num_time_step, num_pitch)
            The merged piano-rolls.
        lowest : int
            Indicate the lowest pitch in the merged piano-roll.
        """
        if not isinstance(mode, str):
            raise TypeError("`mode` must be a string in {'max', 'sum', 'any'}")
        if mode not in ['max', 'sum', 'any']:
            raise TypeError("`mode` must be one of {'max', 'sum', 'any'}")

        stacked, lowest = self.get_stacked_pianorolls(track_indices)

        if mode == 'any':
            merged = np.any(stacked, axis=3)
        elif mode == 'sum':
            merged = np.sum(stacked, axis=3)
            if clipped:
                merged = merged - (merged - upper) * (merged > upper)
        elif mode == 'max':
            merged = np.max(stacked, axis=3)

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
        specified by `track_indices`.

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
                               threshold=0):
        """
        Return a stacked multi-track piano-roll composed of tracks specified
        by `track_indices`. The shape of the return np.ndarray is
        (num_time_step, num_pitch, num_track).

        Notes
        -----
        The ordering of tracks follows what appear in `track_indices`. If
        `track_indices` is None, the original order will be preserved.

        Parameters
        ----------
        track_indices : list
            List of indices that indicate which tracks to be collected. If None
            (by default), all tracks will be collected.
        binarized : bool
            If True, return a binarized stacked piano-rolls. Otherwise, return
            the raw stacked piano-rolls. Default to False.
        threshold : int or float
            Threshold to binarize the collected piano-rolls. Only effective when
            `binarized` is True. Default to zero.

        Returns
        -------
        stacked : np.ndarray, shape=(num_time_step, num_pitch, num_track)
            The stacked piano-roll.
        lowest : int
            Indicate the lowest pitch in the stacked piano-roll.
        """
        if track_indices is None:
            track_indices = range(len(self.tracks))

        lowest, highest = self.get_pitch_range(track_indices)
        length = self.get_length()

        to_stack = []
        for idx in track_indices:
            to_pad_l = self.tracks[idx].lowest - lowest
            to_pad_h = (highest - self.tracks[idx].lowest
                        - self.tracks[idx].pianoroll.shape[1])
            to_pad_t = length - self.tracks[idx].pianoroll.shape[0]
            to_pad = ((0, to_pad_t), (to_pad_l, to_pad_h))
            if binarized:
                binarized = (self.tracks[idx].pianoroll > threshold)
                padded = np.lib.pad(binarized, to_pad, 'constant')
            else:
                padded = np.lib.pad(self.tracks[idx].pianoroll, to_pad,
                                    'constant')
            to_stack.append(padded)

        stacked = np.stack(to_stack, -1)
        return stacked, lowest

    def is_binarized(self, track_indices=None):
        """
        Return True if pianorolls specified by `track_indices` are already
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

    def load(self, filepath_npz, filepath_json):
        """
        Load a previously saved .npz file and JSON file

        Notes
        -----
        Previous values of attributes will all be cleared.

        Parameters
        ----------
        filepath_npz : str
            The path to the .npz file.
        filepath_json : str
            The path to the JSON file.
        """
        def reconstruct_sparse(target_dict, name):
            """
            Return the reconstructed scipy.sparse.csc_matrix, whose components
            are stored in `target_dict` with prefix given as `name`
            """
            return csc_matrix((target_dict[name+'_csc_data'],
                               target_dict[name+'_csc_indices'],
                               target_dict[name+'_csc_indptr']),
                              shape=target_dict[name+'_csc_shape'])

        with open(filepath_json) as infile:
            info_dict = json.load(infile)

        self.name = info_dict['name']
        self.beat_resolution = info_dict['beat_resolution']

        with np.load(filepath_npz) as loaded:
            self.tempo = loaded['tempo']
            if 'downbeat' in loaded.files:
                self.tempo = loaded['downbeat']

            idx = 0
            while str(idx) in info_dict:
                pianoroll = reconstruct_sparse(loaded,
                                               'pianoroll_{}'.format(idx))
                track = Track(pianoroll, info_dict[str(idx)]['program'],
                              info_dict[str(idx)]['is_drum'],
                              info_dict[str(idx)]['name'],
                              info_dict[str(idx)]['lowest'])
                self.tracks.append(track)
                idx += 1

        self.check_validity()

    def merge_tracks(self, track_indices=None, mode='sum', program=0,
                     is_drum=False, name='merged', remove_merged=False,
                     clipped=True, upper=128):
        """
        Merge piano-rolls of tracks specified by `track_indices`. The merged
        track will have program number as given by `program` and drum indicator
        as given by `is_drum`. The merged track will be appended at the end of
        the track list.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to collect. If None (by
            default), all tracks will be collectted.
        mode : {'sum', 'max', 'any'}
            Indicate the merging function to apply along the track axis. Default
            to 'sum'.
            - In 'sum' mode, the piano-roll of the merged track is the summation
              of the collected piano-rolls. Note that for binarized piano-roll,
              integer summation is performed.
            - In 'max' mode, for each pixel, the maximal value among the
              collected piano-rolls is assigned to the merged piano-roll.
            - In 'any' mode, the value of a pixel in the merged piano-roll is
              True if any of the collected piano-rolls has nonzero value at that
              pixel; False if all piano-rolls are inactive (zero-valued) at that
              pixel.
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
        clipped : bool
            True to clip the velocities of the parsed piano-rolls by `upper`.
            False to use the raw sum of the velocities. Only effective in 'sum'
            mode. Default to True.
        upper : int
            The upper bound to clip the input piano-roll. Only effective when
            'clipped' is True. Default to 128.
        """
        if not isinstance(mode, str):
            raise TypeError("`mode` must be a string in {'max', 'sum', 'any'}")
        if mode not in ['max', 'sum', 'any']:
            raise TypeError("`mode` must be one of {'max', 'sum', 'any'}")

        merged, lowest = self.get_merged_pianoroll(track_indices, mode,
                                                   clipped, upper)

        merged_track = Track(merged, program, is_drum, name, lowest)
        self.append_track(merged_track)

        if remove_merged:
            self.remove_tracks(track_indices)

    def parse_midi(self, filepath, mode='sum', algorithm='normal',
                   binarized=False, compressed=True, collect_onsets_only=False,
                   threshold=0, first_beat_time=None):
        """
        Parse a MIDI file

        Parameters
        ----------
        filepath : str
            The path to the MIDI file.
        mode : {'sum', 'max', 'any'}
            Indicate the merging function to apply to duplicate notes. Default
            to 'sum'.
        algorithm : {'normal', 'strict', 'custom'}
            Indicate the method used to get the location of the first beat.
            Notes before it will be dropped unless an incomplete beat before it
            is found (see Notes for details). Default to 'normal'.
            - The 'normal' algorithm estimate the location of the first beat by
            :method:`pretty_midi.PrettyMIDI.estimate_beat_start`.
            - The 'strict' algorithm set the first beat at the event time of the
            first time signature change. If no time signature change event
            found, raise a ValueError.
            - The 'custom' algorithm take argument `first_beat_time` as the
            location of the first beat.
        binarized : bool
            True to binarize the parsed piano-rolls before merging duplicate
            notes. False to use the original parsed piano-rolls. Default to
            False.
        compressed : bool
            True to compress the pitch range of the parsed piano-rolls. False to
            use the original parsed piano-rolls. Deafault to True.
        collect_onsets_only : bool
            True to collect only the onset of the notes (i.e. note on events) in
            all tracks, where the note off and duration information are dropped.
            False to parse regular piano-rolls.
        threshold : int or float
            Threshold to binarize the parsed piano-rolls. Only effective when
            `binarized` is True. Default to zero.
        first_beat_time : float
            The location (in sec) of the first beat. Required and only effective
            when using 'custom' algorithm.

        Returns
        -------
        midi_info : dict
            Contains additional information of the parsed MIDI file as fallows.
            - first_beat_time (float) : the location (in sec) of the first beat
            - incomplete_beat_at_start (bool) : indicate whether there is an
              incomplete beat before `first_beat_time`
            - num_time_signature_change (int) : the number of time signature
              change events
            - time_signature (str) : the time signature (in 'X/X' format) if
              there is only one time signature events. None if no time signature
              event found
            - tempo (float) : the tempo value (in bpm) if there is only one
              tempo change events. None if no tempo change event found

        Notes
        -----
        If an incomplete beat before the first beat is found, an additional beat
        will be added before the (estimated) beat start time. However, notes
        before the (estimated) beat start time for more than one beat are
        dropped.

        Notes
        -----
        If an incomplete beat before the first beat is found, an additional beat
        will be added before the (estimated) beat start time. However, notes
        before the (estimated) beat start time for more than one beat are
        dropped.
        """
        pm = pretty_midi.PrettyMIDI(filepath)
        midi_info = self.parse_pretty_midi(pm, mode, algorithm, binarized,
                                           compressed, collect_onsets_only,
                                           threshold, first_beat_time)
        return midi_info

    def parse_pretty_midi(self, pm, mode='sum', algorithm='normal',
                          binarized=False, compressed=True,
                          collect_onsets_only=False, threshold=0,
                          first_beat_time=None):
        """
        Parse a :class:`pretty_midi.PrettyMIDI` object

        Parameters
        ----------
        pm : `pretty_midi.PrettyMIDI` object
            The :class:`pretty_midi.PrettyMIDI` object to be parsed.
        mode : {'sum', 'max', 'any'}
            Indicate the merging function to apply to duplicate notes. Default
            to 'sum'.
        algorithm : {'normal', 'strict', 'custom'}
            Indicate the method used to get the location of the first beat.
            Notes before it will be dropped unless an incomplete beat before it
            is found (see Notes for details). Default to 'normal'.
            - The 'normal' algorithm estimate the location of the first beat by
            :method:`pretty_midi.PrettyMIDI.estimate_beat_start`.
            - The 'strict' algorithm set the first beat at the event time of the
            first time signature change. If no time signature change event
            found, raise a ValueError.
            - The 'custom' algorithm take argument `first_beat_time` as the
            location of the first beat.
        binarized : bool
            True to binarize the parsed piano-rolls before merging duplicate
            notes. False to use the original parsed piano-rolls. Default to
            False.
        compressed : bool
            True to compress the pitch range of the parsed piano-rolls. False to
            use the original parsed piano-rolls. Deafault to True.
        collect_onsets_only : bool
            True to collect only the onset of the notes (i.e. note on events) in
            all tracks, where the note off and duration information are dropped.
            False to parse regular piano-rolls.
        threshold : int or float
            Threshold to binarize the parsed piano-rolls. Only effective when
            `binarized` is True. Default to zero.
        first_beat_time : float
            The location (in sec) of the first beat. Required and only effective
            when using 'custom' algorithm.

        Returns
        -------
        midi_info : dict
            Contains additional information of the parsed MIDI file as fallows.
            - first_beat_time (float) : the location (in sec) of the first beat
            - incomplete_beat_at_start (bool) : indicate whether there is an
              incomplete beat before `first_beat_time`
            - num_time_signature_change (int) : the number of time signature
              change events
            - time_signature (str) : the time signature (in 'X/X' format) if
              there is only one time signature events. None if no time signature
              event found
            - tempo (float) : the tempo value (in bpm) if there is only one
              tempo change events. None if no tempo change event found

        Notes
        -----
        If an incomplete beat before the first beat is found, an additional beat
        will be added before the (estimated) beat start time. However, notes
        before the (estimated) beat start time for more than one beat are
        dropped.
        """
        if not isinstance(mode, str):
            raise TypeError("`mode` must be a string in {'max', 'sum', 'any'}")
        if mode not in ['max', 'sum', 'any']:
            raise TypeError("`mode` must be one of {'max', 'sum', 'any'}")

        if not isinstance(algorithm, str):
            raise TypeError("`algorithm` must be a string in {'normal', "
                            "'strict', 'custom'}")
        if algorithm not in ['strict', 'normal', 'custom']:
            raise ValueError("`algorithm` must be one of 'normal', 'strict' "
                             "and 'custom'")
        if algorithm == 'custom':
            if not isinstance(first_beat_time, (int, float)):
                raise TypeError("`first_beat_time` must be a number when "
                                "using 'custom' algorithm")
            if first_beat_time < 0.0:
                raise ValueError("`first_beat_time` must be a positive number "
                                 "when using 'custom' algorithm")

        # Set first_beat_time for 'normal' and 'strict' modes
        if mode == 'normal':
            first_beat_time = pm.estimate_beat_start()
        elif mode == 'strict':
            if not pm.time_signature_changes:
                raise ValueError("No time signature change event found. Unable "
                                 "to set beat start time using 'strict' "
                                 "algorithm")
            pm.time_signature_changes.sort(key=lambda x: x.time)
            first_beat_time = pm.time_signature_changes[0].time

        # get tempo change event times and contents
        tc_times, tempi = pm.get_tempo_changes()
        arg_sorted = np.argsort(tc_times.argsort)
        tc_times = tc_times[arg_sorted]
        tempi = tempi[arg_sorted]

        # The following section find the time (`one_beat_ahead`) that is exactly
        # one beat before `first_beat_time`
        # ========= start =========
        remained_beat = 1.0
        one_beat_ahead = first_beat_time
        end = one_beat_ahead

        # Initialize `tc_idx` to the index of the nearest tempo change event
        # before `first_beat_time`
        tc_idx = 0
        while tc_times[tc_idx] > first_beat_time:
            tc_idx += 1
        tc_idx = max(0, tc_idx-1)

        while remained_beat > 0.0:
            # Check if it is the first tempo change event. If so, reduce
            # `one_beat_ahead` by the corresponding duration of `remained_beat`
            # in current tempo and break iteration
            if tc_idx < 1:
                one_beat_ahead -= remained_beat * 60. / tempi[tc_idx]
                break

            # Check if current tempo can fill up `remained_beat`. If so, reduce
            # `one_beat_ahead` by the corresponding duration of `remained_beat`
            # in current tempo and break iteration
            tc_duration_in_beat = (end - tc_times[tc_idx]) * tempi[tc_idx] / 60.
            if remained_beat < tc_duration_in_beat:
                one_beat_ahead -= remained_beat * 60. / tempi[tc_idx]
                break

            # Reduce `one_beat_ahead` by the duration of current tempo
            one_beat_ahead -= (end - tc_times[tc_idx])

            # Reduce `remained_beat` by the duration (in beat) of current tempo
            remained_beat -= tc_duration_in_beat
            end = tc_times[tc_idx]

            tc_idx -= 1
        # ========== end ==========

        # Add an additional beat if any note found between `one_beat_ahead` and
        # `first_beat_time`
        incomplete_beat_found = False
        for instrument in pm.instruments:
            for note in instrument.notes:
                if ((one_beat_ahead < note.start < first_beat_time)
                        or (one_beat_ahead < note.end < first_beat_time)):
                    incomplete_beat_found = True
                    break
        beat_times = pm.get_beats(first_beat_time)
        beat_times.sort()
        num_beat = len(beat_times) + incomplete_beat_found
        num_time_step = self.beat_resolution * num_beat

        # Parse downbeat array
        if not pm.time_signature_changes:
            self.downbeat = None
        else:
            self.downbeat = np.zeros((num_time_step, ), bool)
            self.downbeat[0] = True
            start = int(incomplete_beat_found)
            end = start
            for idx, tsc in enumerate(pm.time_signature_changes[:-1]):
                end += np.searchsorted(beat_times[end:],
                                       pm.time_signature_changes[idx+1])
                start_idx = start * self.beat_resolution
                end_idx = end * self.beat_resolution
                stride = tsc.numerator * self.beat_resolution
                self.downbeat[start_idx:end_idx:stride] = True
                start = end

        # Parse tempo array
        self.tempo = np.empty((num_time_step,))
        if not tc_times:
            estimated_tempo = pm.estimate_tempo()
            self.tempo.fill(estimated_tempo)
        else:
            # here we assume all the tempo events are close to the beats and
            # align time change event to the nearest beat times befrore it
            start = np.searchsorted(beat_times, tc_times[0])
            self.tempo[:start*self.beat_resolution] = tempi[0]
            end = start
            for idx, tempo in enumerate(tempi[:-1]):
                end += np.searchsorted(beat_times[end:], tc_times[idx+1])
                start_idx = start * self.beat_resolution
                end_idx = end * self.beat_resolution
                self.tempo[start_idx:end_idx] = tempo
                start = end
            self.tempo[end_idx:] = tempi[-1]

        # Find the corresponding time in the original MIDI file of each time
        # step in the piano-roll, tempo array and beat array
        if incomplete_beat_found:
            one_beat_ahead = beat_times - (60. / self.tempo[0])
            beat_times = np.insert(beat_times, 0, one_beat_ahead)
        beat_times_tiled = np.tile(beat_times.reshape(-1, 1), (1, 24))
        time_step_durations = (np.reshape(self.tempo, (-1, 24))
                               / (60. * self.beat_resolution))
        time_step_durations[:, 1:] = time_step_durations[:, :-1]
        time_step_durations[:, 0] = 0.
        time_step_times = beat_times_tiled + time_step_durations

        even_step_times = time_step_times[::2]
        odd_step_times = time_step_times[1::2]

        # Parse piano-roll
        piano_roll = None
        for instrument in pm.instruments:
            if piano_roll is None:
                if binarized or mode == 'any':
                    piano_roll = np.zeros((num_time_step, 128), bool)
                else:
                    piano_roll = np.zeros((num_time_step, 128), int)
            else:
                piano_roll.fill(0)

            note_on_times = np.array([note.start for note in instrument.notes
                                      if note.end < one_beat_ahead])
            note_on = 2 * np.searchsorted(even_step_times, note_on_times)

            if collect_onsets_only:
                piano_roll[note_on] = True
            elif instrument.is_drum:
                piano_roll[note_on:] = True
            else:
                note_off_times = np.array([note.end for note in instrument.notes
                                           if note.end < one_beat_ahead])
                note_off = (2 * np.searchsorted(odd_step_times, note_off_times)
                            + 1)

            for idx, start in enumerate(note_on):
                end = note_off[idx]
                velocity = instrument.notes[idx].velocity
                if binarized:
                    if velocity > threshold:
                        if mode == 'sum':
                            piano_roll[start:end] += 1
                        elif mode == 'max' or mode == 'any':
                            piano_roll[start:end] = True
                elif mode == 'sum':
                    piano_roll[start:end] += velocity
                elif mode == 'max':
                    piano_roll[start:end] = np.maximum(piano_roll[start:end],
                                                       velocity)
                elif mode == 'any':
                    if velocity:
                        piano_roll[start:end] = True

            track = Track(piano_roll, instrument.program, instrument.is_drum,
                          instrument.name)
            if compressed:
                track.compress()
            self.tracks.append(track)

        self.check_validity()

        # Collect midi info into a dictionary and return it
        num_ts_change = len(pm.time_signature_changes)
        if num_ts_change == 1:
            time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                       pm.time_signature_changes[0].denominator)
        else:
            time_sign = None

        midi_info = {'first_beat_time': first_beat_time,
                     'incomplete_beat_at_start': incomplete_beat_found,
                     'num_time_signature_change': num_ts_change,
                     'time_signature': time_sign,
                     'tempo': tempi[0] if len(tc_times) == 1 else None}

        return midi_info

    def remove_tracks(self, track_indices):
        """
        Remove tracks specified by `track_indices`.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to remove.
        """
        self.tracks = [track for idx, track in enumerate(self.tracks)
                       if idx not in track_indices]

    def save(self, filepath_npz, filepath_json, track_indices=None,
             compressed=True):
        """
        Save numpy arrays to a (compressed) .npz file and other information to a
        JSON file

        Notes
        -----
        - To reduce the file size, the collected piano-rolls are first converted
          to instances of scipy.sparse.csc_matrix, whose component arrays are
          then collected and saved to the .npz file.
        - The ordering of tracks in the saved .npz file follows what appear in
          `track_indices`. If `track_indices` is None, the original order will
          be preserved.

        Parameters
        ----------
        filepath_npz : str
            The path to write the .npz file.
        filepath_json : str
            The path to write the JSON file.
        track_indices : list
            List of indices that indicates which tracks to save. If None (by
            default), all tracks will be converted.
        compressed : bool
            True to save to a compressed .npz file. False to save to a
            uncompressed .npz file. Default to True.
        """

        def update_sparse(target_dict, sparse_matrix, name):
            """
            Turn `sparse_matrix` into a scipy.sparse.csc_matrix and update its
            component arrays to the `target_dict` with key as `name` postfixed
            with its component type string
            """
            csc = csc_matrix(sparse_matrix)
            target_dict[name+'_csc_data'] = csc.data
            target_dict[name+'_csc_indices'] = csc.indices
            target_dict[name+'_csc_indptr'] = csc.indptr
            target_dict[name+'_csc_shape'] = csc.shape

        if track_indices is None:
            track_indices = range(len(self.tracks))

        array_dict = {'tempo': self.tempo}
        info_dict = {'beat_resolution': self.beat_resolution,
                     'name': self.name}

        if self.downbeat:
            array_dict['downbeat'] = self.downbeat

        for idx, track_idx in enumerate(track_indices):
            update_sparse(array_dict, self.tracks[track_idx].pianoroll,
                          'pianoroll_{}'.format(idx))
            info_dict[str(idx)] = {'program': self.tracks[track_idx].program,
                                   'is_drum': self.tracks[track_idx].is_drum,
                                   'name': self.tracks[track_idx].name,
                                   'lowest': self.tracks[track_idx].lowest}

        if not filepath_npz.endswith('.npz'):
            filepath_npz += '.npz'
        if compressed:
            np.savez_compressed(filepath_npz, **array_dict)
        else:
            np.savez(filepath_npz, **array_dict)

        with open(filepath_json, 'w') as outfile:
            json.dump(info_dict, outfile)

    def to_pretty_midi(self, track_indices=None, tempo=False, downbeat=False):
        """
        Convert to a :class:`pretty_midi.PrettyMIDI` instance

        Notes
        -----
        The velocities of the converted piano-rolls are cliiped into [0, 127],
        i.e. values below 0 and values beyond 127 are replaced by 127 and 0,
        respectively.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to convert. If None (by
            default), all tracks will be converted.

        Returns
        -------
        pm : `pretty_midi.PrettyMIDI` object
            The converted :class:`pretty_midi.PrettyMIDI` instance.
        """

        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo[0])

        if track_indices is None:
            track_indices = range(len(self.tracks))

        # TODO: Add downbeat support -> time signature change events
        # TODO: Add tempo support -> tempo change events
        tempo = self.tempo[0]
        time_step_size = 60. / tempo / self.beat_resolution

        for track_idx in track_indices:
            track = deepcopy(self.tracks[track_idx])
            instrument = pretty_midi.Instrument(program=track.program,
                                                is_drum=track.is_drum,
                                                name=track.name)

            track.clip()
            clipped = track.pianoroll.astype(int)
            binarized = clipped.astype(bool)
            padded = np.pad(binarized, ((1, 1), (0, 0)), 'constant')
            diff = np.diff(padded, axis=0)

            track.get_pitch_range()
            for pitch in range(128):
                note_ons = np.nonzero(diff[:, pitch] > 0)
                note_on_times = time_step_size * note_ons[0]
                note_offs = np.nonzero(diff[:, pitch] < 0)
                note_off_times = time_step_size * note_offs[0]

                for idx, note_on_time in enumerate(note_on_times):
                    velocity = np.mean(clipped[note_ons[idx]:note_offs[idx]])
                    note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                            start=note_on_time,
                                            end=note_off_times[idx])
                    instrument.notes.append(note)

            instrument.notes.sort(key=lambda x: x.start)
            pm.instruments.append(instrument)

        return pm

    def transpose(self, semitone):
        """
        Transpose the piano-rolls by `semitones` semitones

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

    def write_midi(self, filepath, track_indices=None):
        """
        Write to a MIDI file

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to convert. If None
            (by default), all tracks will be collected.

        Parameters
        ----------
        filepath : str
            The path to write the MIDI file.
        """
        pm = self.to_pretty_midi(track_indices)
        pm.write(filepath)

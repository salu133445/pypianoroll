"""Class for multi-track piano-rolls with metadata.

"""
from __future__ import division
import json
from copy import deepcopy
import numpy as np
import pretty_midi
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.sparse import csc_matrix
from .track import Track
from .plot import plot_pianoroll

class Multitrack(object):
    """
    A multi-track piano-roll container

    Attributes
    ----------
    tracks : list
        List of :class:`pypianoroll.Track` objects.
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

    """
    def __init__(self, filepath=None, filepath_json=None, tracks=None,
                 tempo=120.0, downbeat=None, beat_resolution=24,
                 name='unknown'):
        """
        Initialize the object by one of the following ways:
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
            List of :class:`pypianoroll.Track` objects to be added to the track
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
            Name of the multi-track piano-roll. Default to 'unknown'.

        """
        # parse input file
        if filepath is not None:
            if not isinstance(filepath, str):
                raise TypeError("`filepath` must be of str type")
            if filepath.endswith(('.mid', '.midi', '.MID', '.MIDI')):
                self.beat_resolution = beat_resolution
                self.name = name
                self.parse_midi(filepath)
            elif filepath.endswith('.npz'):
                if filepath_json is None:
                    raise TypeError("`filepath_json` must be given when "
                                    "`filepath` is a .npz file")
                self.load(filepath, filepath_json)
            else:
                raise ValueError("Unsupported file type")
        else:
            if tracks is not None:
                self.tracks = tracks
            else:
                self.tracks = [Track()]
            if isinstance(tempo, (int, float)):
                self.tempo = np.array([tempo])
            else:
                self.tempo = tempo
            self.downbeat = downbeat
            self.beat_resolution = beat_resolution
            self.name = name
            self.check_validity()

    def __getitem__(self, val):
        if isinstance(val, tuple):
            if isinstance(val[0], int):
                tracks = self.tracks[val[0]][val[1:]]
            if isinstance(val[0], list):
                tracks = [self.tracks[i][val[1:]] for i in val[0]]
            else:
                tracks = [track[val[1:]] for track in self.tracks[val[0]]]
            return Multitrack(tracks=tracks, tempo=self.tempo[val[1]],
                              downbeat=self.downbeat[val[1]],
                              beat_resolution=self.beat_resolution,
                              name=self.name)
        if isinstance(val, list):
            tracks = [self.tracks[i] for i in val[0]]
        else:
            tracks = self.tracks[val]
        return Multitrack(tracks=tracks, tempo=self.tempo,
                          downbeat=self.downbeat,
                          beat_resolution=self.beat_resolution, name=self.name)

    def __repr__(self):
        track_names = ', '.join([repr(track.name) for track in self.tracks])
        return ("Multitrack(tracks=[{}], tempo={}, downbeat={}, beat_resolution"
                "={}, name={})".format(track_names, repr(self.tempo),
                                       repr(self.downbeat),
                                       self.beat_resolution, self.name))

    def __str__(self):
        track_names = ', '.join([str(track.name) for track in self.tracks])
        return ("tracks : [{}],\ntempo : {},\ndownbeat : {},\nbeat_resolution "
                ": {},\nname : {}".format(track_names, str(self.tempo),
                                          str(self.downbeat),
                                          self.beat_resolution, self.name))

    def append_track(self, track=None, pianoroll=None, program=0, is_drum=False,
                     name='unknown'):
        """
        Append a multitrack.Track instance to the track list or create a new
        multitrack.Track object and append it to the track list.

        Parameters
        ----------
        track : pianoroll.Track
            A :class:`pypianoroll.Track` instance to be appended to the track
            list.
        pianoroll : np.ndarray, shape=(num_time_step, 128)
            Piano-roll matrix. First dimension represents time. Second dimension
            represents pitch. Available datatypes are bool, int, float.
        program: int
            Program number according to General MIDI specification [1].
            Available values are 0 to 127. Default to 0 (Acoustic Grand Piano).
        is_drum : bool
            Drum indicator. True for drums. False for other instruments. Default
            to False.
        name : str
            Name of the track. Default to 'unknown'.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set

        """
        if track is not None:
            if not isinstance(track, Track):
                raise TypeError("`track` must be a multitrack.Track instance")
            track.check_validity()
        else:
            track = Track(pianoroll, program, is_drum, name)
        self.tracks.append(track)

    def binarize(self, threshold=0):
        """
        Binarize the piano-rolls of all tracks. Pass the track if its piano-roll
        is already binarized.

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano-rolls. Default to zero.

        """
        for track in self.tracks:
            track.binarize(threshold)

    def check_validity(self):
        """
        Raise an error if any invalid attribute found.

        Raises
        ------
        TypeError
            If an attribute has an invalid type.
        ValueError
            If an attribute has an invalid value (of the correct type).

        """
        # tracks
        for track in self.tracks:
            if not isinstance(track, Track):
                raise TypeError("`tracks` must be a list of "
                                "`pypianoroll.Track` instances")
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
        if self.beat_resolution%2 > 0:
            raise ValueError("`beat_resolution` must be an even number")
        # name
        if not isinstance(self.name, str):
            raise TypeError("`name` must be of str type")

    def clip(self, lower=0, upper=128):
        """
        Clip the piano-rolls by an lower bound and an upper bound specified by
        `lower` and `upper`, respectively.

        Parameters
        ----------
        lower : int or float
            The lower bound to clip the piano-roll. Default to 0.
        upper : int or float
            The upper bound to clip the piano-roll. Default to 128.

        """
        for track in self.tracks:
            track.clip(lower, upper)

    def copy(self):
        """
        Return a copy of the object.

        Returns
        -------
        copied : `pypianoroll.Multitrack` object
            A copy of the object.

        """
        copied = deepcopy(self)
        return copied

    def pad_to_same(self):
        """
        Pad shorter piano-rolls with zeros at the end along the time axis to the
        length of the piano-roll with the maximal length.

        """
        maximal_length = self.get_maximal_length()
        for track in self.tracks:
            if track.pianoroll.shape[1] < maximal_length:
                track.pad(maximal_length - track.pianoroll.shape[1])

    def get_active_length(self):
        """
        Return the maximal active length (i.e. without trailing silence) of the
        piano-rolls (in time step).

        Returns
        -------
        active_length : int
            The maximal active length (i.e. without trailing silence) of the
            piano-rolls (in time step).

        """
        active_length = 0
        for track in self.tracks:
            now_length = track.get_active_length()
            if active_length < track.get_active_length():
                active_length = now_length
        return active_length

    def get_downbeat_steps(self):
        """
        Return a list of indices of time steps that contain downbeats.

        Returns
        -------
        downbeat_steps : list
            Indices of time steps that contain downbeats.

        """
        if self.downbeat is None:
            return []
        downbeat_steps = np.nonzero(self.downbeat)[0].tolist()
        return downbeat_steps

    def get_empty_tracks(self):
        """
        Return the indices of tracks with empty piano-rolls.

        Returns
        -------
        empty_track_indices : list
            List of the indices of tracks with empty piano-rolls.

        """
        empty_track_indices = [idx for idx, track in enumerate(self.tracks)
                               if not np.any(track.pianoroll)]
        return empty_track_indices

    def get_maximal_length(self):
        """
        Return the maximal length of the piano-rolls along the time axis (in
        time step).

        Returns
        -------
        maximal_length : int
            The maximal length of the piano-rolls along the time axis (in time
            step).

        """
        maximal_length = 0
        for track in self.tracks:
            now_length = track.pianoroll.shape[0]
            if maximal_length < track.pianoroll.shape[0]:
                maximal_length = now_length
        return maximal_length

    def get_merged_pianoroll(self, mode='sum'):
        """
        Return a merged piano-roll.

        Parameters
        ----------
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

        Returns
        -------
        merged : np.ndarray, shape=(num_time_step, 128)
            The merged piano-rolls.

        """
        if not isinstance(mode, str):
            raise TypeError("`mode` must be a string in {'max', 'sum', 'any'}")
        if mode not in ['max', 'sum', 'any']:
            raise TypeError("`mode` must be one of {'max', 'sum', 'any'}")

        stacked = self.get_stacked_pianorolls()

        if mode == 'any':
            merged = np.any(stacked, axis=2)
        elif mode == 'sum':
            merged = np.sum(stacked, axis=2)
        elif mode == 'max':
            merged = np.max(stacked, axis=2)

        return merged

    def get_num_downbeat(self):
        """
        Return the number of down beats. The return value is calculated based
        solely on `downbeat`.

        Returns
        -------
        num_bar : int
            The number of down beats according to `downbeat`.

        """
        num_bar = np.sum(self.downbeat)
        return num_bar

    def get_num_track(self):
        """
        Return the number of tracks.

        Returns
        -------
        num_track : int
            The number of tracks.

        """
        num_track = len(self.tracks)
        return num_track

    def get_active_pitch_range(self):
        """
        Return the overall active pitch range of the piano-rolls.

        Returns
        -------
        lowest : int
            The lowest active pitch of the piano-rolls.
        highest : int
            The lowest highest pitch of the piano-rolls.

        """
        lowest, highest = self.tracks[0].get_active_pitch_range()
        if len(self.tracks) > 1:
            for track in self.tracks[1:]:
                low, high = track.get_active_pitch_range()
                if low < lowest:
                    lowest = low
                if high > highest:
                    highest = high
        return lowest, highest

    def get_stacked_pianorolls(self):
        """
        Return a stacked multi-track piano-roll. The shape of the return
        np.ndarray is (num_time_step, 128, num_track).

        Returns
        -------
        stacked : np.ndarray, shape=(num_time_step, 128, num_track)
            The stacked piano-roll.

        """
        multitrack = deepcopy(self)
        multitrack.pad_to_same()
        stacked = np.stack([track.pianoroll for track in multitrack.tracks], -1)
        return stacked

    def is_binarized(self):
        """
        Return True if the pianorolls of all tracks are already binarized.
        Otherwise, return False.

        Returns
        -------
        is_binarized : bool
            True if all the collected piano-rolls are already binarized;
            otherwise, False.

        """
        for track in self.tracks:
            if not track.is_binarized():
                return False
        return True

    def load(self, filepath_npz, filepath_json):
        """
        Load a previously saved .npz file and JSON file.

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
            are stored in `target_dict` with prefix given as `name`.

            """
            return csc_matrix((target_dict[name+'_csc_data'],
                               target_dict[name+'_csc_indices'],
                               target_dict[name+'_csc_indptr']),
                              shape=target_dict[name+'_csc_shape']).toarray()

        with open(filepath_json) as infile:
            info_dict = json.load(infile)

        self.name = str(info_dict['name'])
        self.beat_resolution = info_dict['beat_resolution']

        with np.load(filepath_npz) as loaded:
            self.tempo = loaded['tempo']
            if 'downbeat' in loaded.files:
                self.downbeat = loaded['downbeat']

            idx = 0
            self.tracks = []
            while str(idx) in info_dict:
                pianoroll = reconstruct_sparse(loaded,
                                               'pianoroll_{}'.format(idx))
                track = Track(pianoroll, info_dict[str(idx)]['program'],
                              info_dict[str(idx)]['is_drum'],
                              str(info_dict[str(idx)]['name']))
                self.tracks.append(track)
                idx += 1

        self.check_validity()

    def merge_tracks(self, track_indices=None, mode='sum', program=0,
                     is_drum=False, name='merged', remove_merged=False):
        """
        Merge piano-rolls of tracks specified by `track_indices`. The merged
        track will have program number as given by `program` and drum indicator
        as given by `is_drum`. The merged track will be appended at the end of
        the track list.

        Parameters
        ----------
        track_indices : list
            List of indices that indicates which tracks to merge. If None,
            default to merge all tracks.
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

        """
        if not isinstance(mode, str):
            raise TypeError("`mode` must be a string in {'max', 'sum', 'any'}")
        if mode not in ['max', 'sum', 'any']:
            raise TypeError("`mode` must be one of {'max', 'sum', 'any'}")

        merged = self[track_indices].get_merged_pianoroll(mode)

        merged_track = Track(merged, program, is_drum, name)
        self.append_track(merged_track)

        if remove_merged:
            self.remove_tracks(track_indices)

    def parse_midi(self, filepath, mode='sum', algorithm='normal',
                   binarized=False, compressed=True, collect_onsets_only=False,
                   collect_midi_info=False, threshold=0, first_beat_time=None):
        """
        Parse a MIDI file.

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
            :meth:`pretty_midi.PrettyMIDI.estimate_beat_start`.
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
        collect_midi_info : bool
            True to collect additional information in the MIDI file and return
            in a dictionary format. False to collect and return nothing.
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
        pm = pretty_midi.PrettyMIDI(filepath)
        midi_info = self.parse_pretty_midi(pm, mode, algorithm, binarized,
                                           compressed, collect_onsets_only,
                                           collect_midi_info, threshold,
                                           first_beat_time)
        return midi_info

    def parse_pretty_midi(self, pm, mode='sum', algorithm='normal',
                          binarized=False, skip_empty_tracks=True,
                          collect_onsets_only=False, collect_midi_info=False,
                          threshold=0, first_beat_time=None):
        """
        Parse a :class:`pretty_midi.PrettyMIDI` object.

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
            :meth:`pretty_midi.PrettyMIDI.estimate_beat_start`.
            - The 'strict' algorithm set the first beat at the event time of the
            first time signature change. If no time signature change event
            found, raise a ValueError.
            - The 'custom' algorithm take argument `first_beat_time` as the
            location of the first beat.
        binarized : bool
            True to binarize the parsed piano-rolls before merging duplicate
            notes. False to use the original parsed piano-rolls. Default to
            False.
        skip_empty_tracks : bool
            True to remove tracks with empty piano-rolls and compress the pitch
            range of the parsed piano-rolls. False to retain the empty tracks
            and use the original parsed piano-rolls. Deafault to True.
        collect_onsets_only : bool
            True to collect only the onset of the notes (i.e. note on events) in
            all tracks, where the note off and duration information are dropped.
            False to parse regular piano-rolls.
        collect_midi_info : bool
            True to collect additional information in the MIDI file and return
            in a dictionary format. False to collect and return nothing.
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
        if algorithm == 'normal':
            if pm.time_signature_changes:
                pm.time_signature_changes.sort(key=lambda x: x.time)
                first_beat_time = pm.time_signature_changes[0].time
            else:
                first_beat_time = pm.estimate_beat_start()
        elif algorithm == 'strict':
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

        beat_times = pm.get_beats(first_beat_time)
        beat_times.sort()
        num_beat = len(beat_times)
        num_time_step = self.beat_resolution * num_beat

        # Parse downbeat array
        if not pm.time_signature_changes:
            self.downbeat = None
        else:
            self.downbeat = np.zeros((num_time_step, ), bool)
            self.downbeat[0] = True
            start = 0
            end = start
            for idx, tsc in enumerate(pm.time_signature_changes[:-1]):
                end += np.searchsorted(beat_times[end:],
                                       pm.time_signature_changes[idx+1])
                start_idx = start * self.beat_resolution
                end_idx = end * self.beat_resolution
                stride = tsc.numerator * self.beat_resolution
                self.downbeat[start_idx:end_idx:stride] = True
                start = end

        # Build tempo array
        one_more_beat = 2 * beat_times[-1] - beat_times[-2]
        beat_times_one_more = np.append(beat_times, one_more_beat)
        bpm = 60. / np.diff(beat_times_one_more)
        self.tempo = np.tile(bpm, (1, 24)).reshape(-1,)

        # Parse piano-roll
        self.tracks = []
        for instrument in pm.instruments:
            if binarized or mode == 'any':
                pianoroll = np.zeros((num_time_step, 128), bool)
            else:
                pianoroll = np.zeros((num_time_step, 128), int)

            pitches = np.array([note.pitch for note in instrument.notes
                                if note.end > first_beat_time])
            note_on_times = np.array([note.start for note in instrument.notes
                                      if note.end > first_beat_time])
            # note_ons = np.searchsorted(tstep_times, note_on_times)
            beat_indices = np.searchsorted(beat_times, note_on_times) - 1
            remained = note_on_times - beat_times[beat_indices]
            ratios = remained / (beat_times_one_more[beat_indices + 1]
                                 - beat_times[beat_indices])
            note_ons = ((beat_indices + ratios)
                        * self.beat_resolution).astype(int)

            if collect_onsets_only:
                pianoroll[note_ons, pitches] = True
            elif instrument.is_drum:
                if binarized:
                    pianoroll[note_ons, pitches] = True
                else:
                    velocities = [note.velocity for note in instrument.notes
                                  if note.end > first_beat_time]
                    pianoroll[note_ons, pitches] = velocities
            else:
                note_off_times = np.array([note.end for note in instrument.notes
                                           if note.end > first_beat_time])
                # note_offs = np.searchsorted(tstep_times, note_off_times) + 1
                beat_indices = np.searchsorted(beat_times, note_off_times) - 1
                remained = note_off_times - beat_times[beat_indices]
                ratios = remained / (beat_times_one_more[beat_indices + 1]
                                     - beat_times[beat_indices])
                note_offs = ((beat_indices + ratios)
                             * self.beat_resolution).astype(int)

                for idx, start in enumerate(note_ons):
                    end = note_offs[idx] + 1
                    velocity = instrument.notes[idx].velocity

                    if velocity < 1 or (binarized and velocity <= threshold):
                        continue

                    if start > 0:
                        if pianoroll[start - 1, pitches[idx]]:
                            pianoroll[start - 1, pitches[idx]] = 0
                    if end < num_time_step - 1:
                        if pianoroll[end + 1, pitches[idx]]:
                            end -= 1

                    if binarized:
                        if mode == 'sum':
                            pianoroll[start:end, pitches[idx]] += 1
                        elif mode == 'max' or mode == 'any':
                            pianoroll[start:end, pitches[idx]] = True
                    elif mode == 'sum':
                        pianoroll[start:end, pitches[idx]] += velocity
                    elif mode == 'max':
                        maximum = np.maximum(pianoroll[start:end], velocity)
                        pianoroll[start:end, pitches[idx]] = maximum
                    elif mode == 'any':
                        if velocity:
                            pianoroll[start:end, pitches[idx]] = True

            if skip_empty_tracks and not np.any(pianoroll):
                continue

            track = Track(pianoroll, instrument.program, instrument.is_drum,
                          str(instrument.name))
            self.tracks.append(track)

        self.check_validity()

        if not collect_midi_info:
            return

        # Collect midi info into a dictionary and return it
        num_ts_change = len(pm.time_signature_changes)
        if num_ts_change == 1:
            time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                       pm.time_signature_changes[0].denominator)
        else:
            time_sign = None

        midi_info = {'first_beat_time': first_beat_time,
                     'num_time_signature_change': num_ts_change,
                     'time_signature': time_sign,
                     'tempo': tempi[0] if len(tc_times) == 1 else None}

        return midi_info

    def plot(self, filepath=None, mode='separate', track_label='name',
             normalization='standard', preset='default', cmaps=None,
             tick_loc=None, xtick='auto', ytick='octave', xticklabel='on',
             yticklabel='auto', direction='in', label='both', grid='both',
             grid_linestyle=':', grid_linewidth=.5):
        """
        Plot the piano-rolls or save a plot of them.

        Parameters
        ----------
        filepath : str
            The filepath to save the plot. If None, default to save nothing.
        mode : {'separate', 'stacked', 'hybrid'}
            Plotting modes. Default to 'separate'.
            - In 'separate' mode, all the tracks are plotted separately.
            - In 'stacked' mode, a color is assigned based on `cmaps` to the
            piano-roll of each track and the piano-rolls are stacked and
            plotted as a colored image with RGB channels.
            - In 'hybrid' mode, the drum tracks are merged into a 'Drums' track,
            while the other tracks are merged into an 'Others' track, and the
            two merged tracks are then plotted separately.
        track_label : {'name', 'program', 'family', 'off'}
            Add track name, program name, instrument family name or none as
            labels to the track. When `mode` is 'hybrid', all options other
            than 'off' will label the two track with 'Drums' and 'Others'.
        normalization : {'standard', 'auto', 'none'}
            The normalization method to apply to the piano-roll. Default to
            'standard'. If `pianoroll` is binarized, use 'none' anyway.
            - For 'standard' normalization, the normalized values are given by
            N = P / 128, where P, N is the original and normalized piano-roll,
            respectively
            - For 'auto' normalization, the normalized values are given by
            N = (P - m) / (M - m), where P, N is the original and normalized
            piano-roll, respectively, and M, m is the maximum and minimum of the
            original piano-roll, respectively.
            - If 'none', no normalization will be applied to the piano-roll. In
            this case, the values of `pianoroll` should be in [0, 1] in order to
            plot it correctly.
        preset : {'default', 'plain', 'frame'}
            Preset themes for the plot.
            - In 'default' preset, the ticks, grid and labels are on.
            - In 'frame' preset, the ticks and grid are both off.
            - In 'plain' preset, the x- and y-axis are both off.
        cmaps :  tuple or list
            List of `matplotlib.colors.Colormap` instances or colormap codes.
            - When `mode` is 'separate', each element will be passed to each
            call of :func:`matplotlib.pyplot.imshow`. Default to ('Blues',
            'Oranges', 'Greens', 'Reds', 'Purples', 'Greys').
            - when `mode` is stacked, a color is assigned based on `cmaps` to
            the piano-roll of each track. Default to ('hsv').
            - When `mode` is 'hybrid', the first (second) element is used in the
            'Drums' ('Others') track. Default to ('Blues', 'Greens').
        tick_loc : tuple or list
            List of locations to put ticks. Availables elements are 'bottom',
            'top', 'left' and 'right'. If None, default to ('bottom', 'left').
        xtick : {'auto', 'beat', 'step', 'off'}
            Use beat number or step number as ticks along the x-axis, or
            automatically set to 'beat' when `beat_resolution` is given and set
            to 'step', otherwise. Default to 'auto'.
        ytick : {'octave', 'pitch', 'off'}
            Use octave or pitch as ticks along the y-axis. Default to 'octave'.
        xticklabel : {'on', 'off'}
            Indicate whether to add tick labels along the x-axis. Only effective
            when `xtick` is not 'off'.
        yticklabel : {'auto', 'name', 'number', 'off'}
            If 'name', use octave name and pitch name (key name when `is_drum`
            is True) as tick labels along the y-axis. If 'number', use pitch
            number. If 'auto', set to 'name' when `ytick` is 'octave' and
            'number' when `ytick` is 'pitch'. Default to 'auto'. Only effective
            when `ytick` is not 'off'.
        direction : {'in', 'out', 'inout'}
            Put ticks inside the axes, outside the axes, or both. Default to
            'in'. Only effective when `xtick` and `ytick` are not both 'off'.
        label : {'x', 'y', 'both', 'off'}
            Add label to the x-axis, y-axis, both or neither. Default to 'both'.
        grid : {'x', 'y', 'both', 'off'}
            Add grid to the x-axis, y-axis, both or neither. Default to 'both'.
        grid_linestyle : str
            Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linestyle'
            argument.
        grid_linewidth : float
            Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linewidth'
            argument.

        Returns
        -------
        fig : `matplotlib.figure.Figure` object
            A :class:`matplotlib.figure.Figure` object.
        axs : list
            List of :class:`matplotlib.axes.Axes` object.

        """
        def get_track_label(track_label, track=None):
            """Convenient function to get track labels"""
            if track_label == 'name':
                return track.name
            elif track_label == 'program':
                return pretty_midi.program_to_instrument_name(track.program)
            elif track_label == 'family':
                return pretty_midi.program_to_instrument_class(track.program)
            elif track is None:
                return track_label

        def add_tracklabel(ax, track_label, track=None):
            """Convenient function for adding track labels"""
            if not ax.get_ylabel():
                return
            ax.set_ylabel(get_track_label(track_label, track) + '\n\n'
                          + ax.get_ylabel())

        if not self.tracks:
            raise ValueError("There is no track to plot")
        if mode not in ('separate', 'stacked', 'hybrid'):
            raise ValueError("`mode` must be one of {'separate', 'stacked', "
                             "'hybrid'}")
        if track_label not in ('name', 'program', 'family', 'off'):
            raise ValueError("`track_label` must be one of {'name', 'program', "
                             "'family'}")

        if self.is_binarized():
            normalization = 'none'

        if cmaps is None:
            if mode == 'separate':
                cmaps = ('Blues', 'Oranges', 'Greens', 'Reds', 'Purples',
                         'Greys')
            elif mode == 'stacked':
                cmaps = ('hsv')
            else:
                cmaps = ('Blues', 'Greens')

        num_track = len(self.tracks)
        downbeats = self.get_downbeat_steps()

        if mode == 'separate':
            if num_track > 1:
                fig, axs = plt.subplots(num_track, sharex=True)
            else:
                fig, ax = plt.subplots()
                axs = [ax]

            for idx, track in enumerate(self.tracks):
                now_xticklabel = xticklabel if idx < num_track else 'off'
                plot_pianoroll(axs[idx], track.pianoroll, False,
                               self.beat_resolution, downbeats,
                               cmap=cmaps[idx%len(cmaps)],
                               normalization=normalization, preset=preset,
                               tick_loc=tick_loc, xtick=xtick, ytick=ytick,
                               xticklabel=now_xticklabel, yticklabel=yticklabel,
                               direction=direction, label=label, grid=grid,
                               grid_linestyle=grid_linestyle,
                               grid_linewidth=grid_linewidth)
                if track_label != 'none':
                    add_tracklabel(axs[idx], track_label, track)

            if num_track > 1:
                fig.subplots_adjust(hspace=0)

            returns = (fig, axs)

        elif mode == 'stacked':
            is_all_drum = True
            for track in self.tracks:
                if not track.is_drum:
                    is_all_drum = False

            fig, ax = plt.subplots()
            stacked = self.get_stacked_pianorolls()

            colormap = matplotlib.cm.get_cmap(cmaps[0])
            cmatrix = colormap(np.arange(0, 1, 1 / num_track))[:, :3]
            recolored = np.matmul(stacked.reshape(-1, num_track), cmatrix)
            stacked = recolored.reshape(stacked.shape[:2] + (3, ))

            plot_pianoroll(ax, stacked, is_all_drum, self.beat_resolution,
                           downbeats, normalization=normalization,
                           preset=preset, xtick=xtick, ytick=ytick,
                           xticklabel=xticklabel, yticklabel=yticklabel,
                           direction=direction, label=label, grid=grid,
                           grid_linestyle=grid_linestyle,
                           grid_linewidth=grid_linewidth)

            if track_label != 'none':
                patches = [Patch(color=cmatrix[idx],
                                 label=get_track_label(track_label, track))
                           for idx, track in enumerate(self.tracks)]
                plt.legend(handles=patches)

            returns = (fig, [ax])

        elif mode == 'hybrid':
            drums = [i for i, track in enumerate(self.tracks) if track.is_drum]
            others = [i for i in range(len(self.tracks)) if i not in drums]
            merged_drums = self.get_merged_pianoroll(drums)
            merged_others = self.get_merged_pianoroll(others)

            fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
            plot_pianoroll(ax1, merged_drums, True, self.beat_resolution,
                           downbeats, cmap=cmaps[0],
                           normalization=normalization, preset=preset,
                           xtick=xtick, ytick=ytick, xticklabel=xticklabel,
                           yticklabel=yticklabel, direction=direction,
                           label=label, grid=grid,
                           grid_linestyle=grid_linestyle,
                           grid_linewidth=grid_linewidth)
            plot_pianoroll(ax2, merged_others, False, self.beat_resolution,
                           downbeats, cmap=cmaps[1],
                           normalization=normalization, preset=preset,
                           xtick=xtick, ytick=ytick, xticklabel=xticklabel,
                           yticklabel=yticklabel, direction=direction,
                           label=label, grid=grid,
                           grid_linestyle=grid_linestyle,
                           grid_linewidth=grid_linewidth)
            fig.subplots_adjust(hspace=0)

            if track_label != 'none':
                add_tracklabel(ax1, 'Drums')
                add_tracklabel(ax2, 'Others')

            returns = (fig, [ax1, ax2])

        if filepath is not None:
            plt.savefig(filepath)

        return returns

    def remove_empty_tracks(self):
        """Remove tracks with empty piano-rolls."""
        empty_tracks = self.get_empty_tracks()
        self.remove_tracks(empty_tracks)

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

    def save(self, filepath_npz, filepath_json, compressed=True):
        """
        Save numpy arrays to a (compressed) .npz file and other information to a
        JSON file.

        Notes
        -----
        - To reduce the file size, the collected piano-rolls are first converted
          to instances of scipy.sparse.csc_matrix, whose component arrays are
          then collected and saved to the .npz file.

        Parameters
        ----------
        filepath_npz : str
            The path to write the .npz file.
        filepath_json : str
            The path to write the JSON file.
        compressed : bool
            True to save to a compressed .npz file. False to save to a
            uncompressed .npz file. Default to True.

        """

        def update_sparse(target_dict, sparse_matrix, name):
            """
            Turn `sparse_matrix` into a scipy.sparse.csc_matrix and update its
            component arrays to the `target_dict` with key as `name` postfixed
            with its component type string.

            """
            csc = csc_matrix(sparse_matrix)
            target_dict[name+'_csc_data'] = csc.data
            target_dict[name+'_csc_indices'] = csc.indices
            target_dict[name+'_csc_indptr'] = csc.indptr
            target_dict[name+'_csc_shape'] = csc.shape

        array_dict = {'tempo': self.tempo}
        info_dict = {'beat_resolution': self.beat_resolution,
                     'name': self.name}

        if self.downbeat.any():
            array_dict['downbeat'] = self.downbeat

        for idx, track in enumerate(self.tracks):
            update_sparse(array_dict, track.pianoroll,
                          'pianoroll_{}'.format(idx))
            info_dict[str(idx)] = {'program': track.program,
                                   'is_drum': track.is_drum,
                                   'name': track.name}

        if not filepath_npz.endswith('.npz'):
            filepath_npz += '.npz'
        if compressed:
            np.savez_compressed(filepath_npz, **array_dict)
        else:
            np.savez(filepath_npz, **array_dict)

        with open(filepath_json, 'w') as outfile:
            json.dump(info_dict, outfile)

    def to_pretty_midi(self, constant_tempo=None):
        """
        Convert to a :class:`pretty_midi.PrettyMIDI` instance.

        Notes
        -----
        - Only constant tempo is supported by now.
        - The velocities of the converted piano-rolls are cliiped into [0, 127],
          i.e. values below 0 and values beyond 127 are replaced by 127 and 0,
          respectively.

        Parameters
        ----------
        constant_tempo : int
            The constant tempo value of the output object. If None, default to
            use the first element of `tempo`

        Returns
        -------
        pm : `pretty_midi.PrettyMIDI` object
            The converted :class:`pretty_midi.PrettyMIDI` instance.

        """
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo[0])

        # TODO: Add downbeat support -> time signature change events
        # TODO: Add tempo support -> tempo change events
        if constant_tempo is None:
            constant_tempo = self.tempo[0]
        time_step_size = 60. / constant_tempo / self.beat_resolution

        for track in self.tracks:
            instrument = pretty_midi.Instrument(program=track.program,
                                                is_drum=track.is_drum,
                                                name=track.name)
            track = track.copy().clip()
            clipped = track.pianoroll.astype(int)
            binarized = clipped.astype(bool)
            padded = np.pad(binarized, ((1, 1), (0, 0)), 'constant')
            diff = np.diff(padded, axis=0)

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
        Transpose the piano-rolls by `semitones` semitones.

        Parameters
        ----------
        semitone : int
            Number of semitones transpose the piano-rolls.

        """
        for track in self.tracks():
            track.transpose(semitone)

    def trim_trailing_silence(self):
        """Trim the trailing silence of the piano-roll."""
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

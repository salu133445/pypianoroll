"""Class for multitrack pianorolls with metadata.

"""
from __future__ import absolute_import, division, print_function
import json
import zipfile
from copy import deepcopy
from six import string_types
import numpy as np
from scipy.sparse import csc_matrix
import pretty_midi
from pypianoroll.track import Track
from pypianoroll.plot import plot_multitrack

class Multitrack(object):
    """
    A multitrack pianoroll container that stores additional tempo and
    program information along with the pianoroll matrices.

    Attributes
    ----------
    tracks : list of :class:`pypianoroll.Track` objects
        The track object list.
    tempo : np.ndarray, shape=(n_time_steps,), dtype=float
        An array that indicates the tempo value (in bpm) at each time step.
        The length is the total number of time steps.
    downbeat : np.ndarray, shape=(n_time_steps,), dtype=bool
        An array that indicates whether the time step contains a downbeat
        (i.e., the first time step of a bar). The length is the total number
        of time steps.
    beat_resolution : int
        The number of time steps used to represent a beat.
    name : str
        The name of the multitrack pianoroll.

    """
    def __init__(self, filename=None, tracks=None, tempo=120.0, downbeat=None,
                 beat_resolution=24, name='unknown'):
        """
        Initialize the object by one of the following ways:
        - assigning values for attributes
        - loading a npz file
        - parsing a MIDI file

        Notes
        -----
        When `filename` is given, the arguments `tracks`, `tempo`, `downbeat`
        and `name` are ignored.

        Parameters
        ----------
        filename : str
            The file path to the npz file to be loaded or the MIDI file (.mid,
            .midi, .MID, .MIDI) to be parsed.
        tracks : list of :class:`pypianoroll.Track` objects
            The track object list to be assigned to `tracks` when `filename` is
            not provided.
        tempo : int or np.ndarray, shape=(n_time_steps,), dtype=float
            An array that indicates the tempo value (in bpm) at each time step.
            The length is the total number of time steps. Will be assigned to
            `tempo` when `filename` is not provided.
        downbeat : list or np.ndarray, shape=(n_time_steps,), dtype=bool
            An array that indicates whether the time step contains a downbeat
            (i.e., the first time step of a bar). The length is the total number
            of time steps.  Will be assigned to `downbeat` when `filename` is
            not given. If a list of indices is provided, it will be viewed as
            the time step indices of the down beats and converted to a numpy
            array. Default is None.
        beat_resolution : int
            The number of time steps used to represent a beat. Will be assigned
            to `beat_resolution` when `filename` is not provided. Defaults to
            24.
        name : str
            The name of the multitrack pianoroll. Defaults to 'unknown'.

        """
        # parse input file
        if filename is not None:
            if filename.endswith(('.mid', '.midi', '.MID', '.MIDI')):
                self.beat_resolution = beat_resolution
                self.name = name
                self.parse_midi(filename)
            elif filename.endswith('.npz'):
                self.load(filename)
            else:
                raise ValueError("Unsupported file type received. Expect a npz "
                                 "or MIDI file.")
        else:
            if tracks is not None:
                self.tracks = tracks
            else:
                self.tracks = [Track()]
            if isinstance(tempo, (int, float)):
                self.tempo = np.array([tempo])
            else:
                self.tempo = tempo
            if isinstance(downbeat, list):
                self.downbeat = np.zeros((max(downbeat) + 1,), bool)
                self.downbeat[downbeat] = True
            else:
                self.downbeat = downbeat
            self.beat_resolution = beat_resolution
            self.name = name
            self.check_validity()

    def __getitem__(self, val):
        if isinstance(val, tuple):
            if isinstance(val[0], int):
                tracks = [self.tracks[val[0]][val[1:]]]
            elif isinstance(val[0], list):
                tracks = [self.tracks[i][val[1:]] for i in val[0]]
            else:
                tracks = [track[val[1:]] for track in self.tracks[val[0]]]
            if self.downbeat is not None:
                downbeat = self.downbeat[val[1]]
            else:
                downbeat = None
            return Multitrack(
                tracks=tracks, tempo=self.tempo, downbeat=downbeat,
                beat_resolution=self.beat_resolution, name=self.name)
        if isinstance(val, list):
            tracks = [self.tracks[i] for i in val]
        else:
            tracks = self.tracks[val]
        return Multitrack(
            tracks=tracks, tempo=self.tempo, downbeat=self.downbeat,
            beat_resolution=self.beat_resolution, name=self.name)

    def __repr__(self):
        track_names = ', '.join([repr(track.name) for track in self.tracks])
        return ("Multitrack(tracks=[{}], tempo={}, downbeat={}, beat_resolution"
                "={}, name={})".format(
                    track_names, repr(self.tempo), repr(self.downbeat),
                    self.beat_resolution, self.name))

    def __str__(self):
        track_names = ', '.join([str(track.name) for track in self.tracks])
        return ("tracks : [{}],\ntempo : {},\ndownbeat : {},\nbeat_resolution "
                ": {},\nname : {}".format(
                    track_names, str(self.tempo), str(self.downbeat),
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
        pianoroll : np.ndarray, shape=(n_time_steps, 128)
            A pianoroll matrix. The first and second dimension represents time
            and pitch, respectively. Available datatypes are bool, int and
            float. Only effective when `track` is None.
        program: int
            A program number according to General MIDI specification [1].
            Available values are 0 to 127. Defaults to 0 (Acoustic Grand Piano).
            Only effective when `track` is None.
        is_drum : bool
            A boolean number that indicates whether it is a percussion track.
            Defaults to False. Only effective when `track` is None.
        name : str
            The name of the track. Defaults to 'unknown'. Only effective when
            `track` is None.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set

        """
        if track is not None:
            if not isinstance(track, Track):
                raise TypeError("`track` must be a pypianoroll.Track instance.")
            track.check_validity()
        else:
            track = Track(pianoroll, program, is_drum, name)
        self.tracks.append(track)

    def assign_constant(self, value):
        """
        Assign a constant value to the nonzeros in the pianorolls of all tracks.
        If a pianoroll is not binarized, its data type will be preserved. If a
        pianoroll is binarized, it will be casted to the type of `value`.

        Arguments
        ---------
        value : int or float
            The constant value to be assigned to the nonzeros of the pianorolls.

        """
        for track in self.tracks:
            track.assign_constant(value)

    def binarize(self, threshold=0):
        """
        Binarize the pianorolls of all tracks.

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the pianorolls. Defaults to zero.

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
                                "`pypianoroll.Track` instances.")
            track.check_validity()
        # tempo
        if not isinstance(self.tempo, np.ndarray):
            raise TypeError("`tempo` must be int or a numpy array.")
        elif not np.issubdtype(self.tempo.dtype, np.number):
            raise TypeError("Data type of `tempo` must be a subdtype of "
                            "np.number.")
        elif self.tempo.ndim != 1:
            raise ValueError("`tempo` must be a 1D numpy array.")
        if np.any(self.tempo <= 0.0):
            raise ValueError("`tempo` should contain only positive numbers.")
        # downbeat
        if self.downbeat is not None:
            if not isinstance(self.downbeat, np.ndarray):
                raise TypeError("`downbeat` must be a numpy array.")
            if not np.issubdtype(self.downbeat.dtype, np.bool_):
                raise TypeError("Data type of `downbeat` must be bool.")
            if self.downbeat.ndim != 1:
                raise ValueError("`downbeat` must be a 1D numpy array.")
        # beat_resolution
        if not isinstance(self.beat_resolution, int):
            raise TypeError("`beat_resolution` must be int.")
        if self.beat_resolution < 1:
            raise ValueError("`beat_resolution` must be a positive integer.")
        # name
        if not isinstance(self.name, string_types):
            raise TypeError("`name` must be a string.")

    def clip(self, lower=0, upper=127):
        """
        Clip the pianorolls of all tracks by the given lower and upper bounds.

        Parameters
        ----------
        lower : int or float
            The lower bound to clip the pianorolls. Defaults to 0.
        upper : int or float
            The upper bound to clip the pianorolls. Defaults to 127.

        """
        for track in self.tracks:
            track.clip(lower, upper)

    def copy(self):
        """"
        Return a copy of the object.

        Returns
        -------
        copied : `pypianoroll.Multitrack` object
            A copy of the object.

        """
        return deepcopy(self)

    def downsample(self, factor):
        """"
        Downsample the pianorolls of all tracks by the given factor.

        Parameters
        ----------
        factor : int
            The ratio between the original beat resolution and the desired beat
            resolution.

        """
        if self.beat_resolution % factor == 0:
            raise ValueError("Downsample factor must be a factor of the beat "
                             "resolution.")
        self.beat_resolution = self.beat_resolution // factor
        for track in self.tracks:
            track.pianoroll = track.pianoroll[::factor]


    def get_active_length(self):
        """
        Return the maximum active length (i.e., without trailing silence) among
        the pianorolls of all tracks. The unit is time step.

        Returns
        -------
        active_length : int
            The maximum active length (i.e., without trailing silence) among the
            pianorolls of all tracks. The unit is time step.

        """
        active_length = 0
        for track in self.tracks:
            now_length = track.get_active_length()
            if active_length < track.get_active_length():
                active_length = now_length
        return active_length

    def get_active_pitch_range(self):
        """
        Return the active pitch range of the pianorolls of all tracks as a tuple
        (lowest, highest).

        Returns
        -------
        lowest : int
            The lowest active pitch among the pianorolls of all tracks.
        highest : int
            The lowest highest pitch among the pianorolls of all tracks.

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

    def get_downbeat_steps(self):
        """
        Return the indices of time steps that contain downbeats.

        Returns
        -------
        downbeat_steps : list
            The indices of time steps that contain downbeats.

        """
        if self.downbeat is None:
            return []
        downbeat_steps = np.nonzero(self.downbeat)[0].tolist()
        return downbeat_steps

    def get_empty_tracks(self):
        """
        Return the indices of tracks with empty pianorolls.

        Returns
        -------
        empty_track_indices : list
            The indices of tracks with empty pianorolls.

        """
        empty_track_indices = [idx for idx, track in enumerate(self.tracks)
                               if not np.any(track.pianoroll)]
        return empty_track_indices

    def get_max_length(self):
        """
        Return the maximum length of the pianorolls along the time axis (in
        time step).

        Returns
        -------
        max_length : int
            The maximum length of the pianorolls along the time axis (in time
            step).

        """
        max_length = 0
        for track in self.tracks:
            if max_length < track.pianoroll.shape[0]:
                max_length = track.pianoroll.shape[0]
        return max_length

    def get_merged_pianoroll(self, mode='sum'):
        """
        Return the merged pianoroll.

        Parameters
        ----------
        mode : {'sum', 'max', 'any'}
            A string that indicates the merging strategy to apply along the
            track axis. Default to 'sum'.

            - In 'sum' mode, the merged pianoroll is the sum of all the
              pianorolls. Note that for binarized pianorolls, integer summation
              is performed.
            - In 'max' mode, for each pixel, the maximum value among all the
              pianorolls is assigned to the merged pianoroll.
            - In 'any' mode, the value of a pixel in the merged pianoroll is
              True if any of the pianorolls has nonzero value at that pixel;
              False if all pianorolls are inactive (zero-valued) at that pixel.

        Returns
        -------
        merged : np.ndarray, shape=(n_time_steps, 128)
            The merged pianoroll.

        """
        stacked = self.get_stacked_pianoroll()

        if mode == 'any':
            merged = np.any(stacked, axis=2)
        elif mode == 'sum':
            merged = np.sum(stacked, axis=2)
        elif mode == 'max':
            merged = np.max(stacked, axis=2)
        else:
            raise ValueError("`mode` must be one of {'max', 'sum', 'any'}.")

        return merged

    def count_downbeat(self):
        """Return the number of down beats. The return value is calculated based
        solely on `downbeat`."""
        return len(np.nonzero(self.downbeat)[0])

    def get_stacked_pianoroll(self):
        """
        Return a stacked multitrack pianoroll. The shape of the return array is
        (n_time_steps, 128, n_tracks).

        Returns
        -------
        stacked : np.ndarray, shape=(n_time_steps, 128, n_tracks)
            The stacked pianoroll.

        """
        multitrack = deepcopy(self)
        multitrack.pad_to_same()
        stacked = np.stack([track.pianoroll for track in multitrack.tracks], -1)
        return stacked

    def is_binarized(self):
        """Return True if the pianorolls of all tracks are already binarized.
        Otherwise, return False."""
        for track in self.tracks:
            if not track.is_binarized():
                return False
        return True

    def load(self, filename):
        """
        Load a npz file. Supports only files previously saved by
        :meth:`pypianoroll.Multitrack.save`.

        Notes
        -----
        Attribute values will all be overwritten.

        Parameters
        ----------
        filename : str
            The name of the npz file to be loaded.

        """
        def reconstruct_sparse(target_dict, name):
            """Return a reconstructed instance of `scipy.sparse.csc_matrix`."""
            return csc_matrix((target_dict[name+'_csc_data'],
                               target_dict[name+'_csc_indices'],
                               target_dict[name+'_csc_indptr']),
                              shape=target_dict[name+'_csc_shape']).toarray()

        with np.load(filename) as loaded:
            if 'info.json' not in loaded:
                raise ValueError("Cannot find 'info.json' in the npz file.")
            info_dict = json.loads(loaded['info.json'].decode('utf-8'))
            self.name = info_dict['name']
            self.beat_resolution = info_dict['beat_resolution']

            self.tempo = loaded['tempo']
            if 'downbeat' in loaded.files:
                self.downbeat = loaded['downbeat']
            else:
                self.downbeat = None

            idx = 0
            self.tracks = []
            while str(idx) in info_dict:
                pianoroll = reconstruct_sparse(
                    loaded, 'pianoroll_{}'.format(idx))
                track = Track(pianoroll, info_dict[str(idx)]['program'],
                              info_dict[str(idx)]['is_drum'],
                              info_dict[str(idx)]['name'])
                self.tracks.append(track)
                idx += 1

        self.check_validity()

    def merge_tracks(self, track_indices=None, mode='sum', program=0,
                     is_drum=False, name='merged', remove_merged=False):
        """
        Merge pianorolls of the tracks specified by `track_indices`. The merged
        track will have program number as given by `program` and drum indicator
        as given by `is_drum`. The merged track will be appended at the end of
        the track list.

        Parameters
        ----------
        track_indices : list
            The indices of tracks to be merged. Defaults to all the tracks.
        mode : {'sum', 'max', 'any'}
            A string that indicates the merging strategy to apply along the
            track axis. Default to 'sum'.

            - In 'sum' mode, the merged pianoroll is the sum of the collected
              pianorolls. Note that for binarized pianorolls, integer summation
              is performed.
            - In 'max' mode, for each pixel, the maximum value among the
              collected pianorolls is assigned to the merged pianoroll.
            - In 'any' mode, the value of a pixel in the merged pianoroll is
              True if any of the collected pianorolls has nonzero value at that
              pixel; False if all the collected pianorolls are inactive
              (zero-valued) at that pixel.

        program: int
            A program number according to General MIDI specification [1].
            Available values are 0 to 127. Defaults to 0 (Acoustic Grand Piano).
        is_drum : bool
            A boolean number that indicates whether it is a percussion track.
            Defaults to False.
        name : str
            A name to be assigned to the merged track. Defaults to 'merged'.
        remove_merged : bool
            True to remove the source tracks from the track list. False to keep
            them. Defaults to False.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set

        """
        if mode not in ('max', 'sum', 'any'):
            raise ValueError("`mode` must be one of {'max', 'sum', 'any'}.")

        merged = self[track_indices].get_merged_pianoroll(mode)

        merged_track = Track(merged, program, is_drum, name)
        self.append_track(merged_track)

        if remove_merged:
            self.remove_tracks(track_indices)

    def pad(self, pad_length):
        """
        Pad the pianorolls of all tracks with zeros at the end along the time
        axis.

        Notes
        -----
        The resulting pianoroll lengths are not guaranteed to be the same. See
        :meth:`pypianoroll.Multitrack.pad_to_same`.

        Parameters
        ----------
        pad_length : int
            The length to pad along the time axis with zeros.

        """
        for track in self.tracks:
            track.pad(pad_length)

    def pad_to_multiple(self, factor):
        """
        Pad the pianoroll of each track with zeros at the end along the time
        axis with the minimum length that makes the resulting pianoroll length a
        multiple of `factor`. Do nothing if the length of a pianoroll is already
        a multiple of `factor`.

        Notes
        -----
        The resulting pianoroll lengths are not guaranteed to be the same. See
        :meth:`pypianoroll.Multitrack.pad_to_same`.

        Parameters
        ----------
        factor : int
            The value which the length of the resulting pianorolls will be a
            multiple of.

        """
        for track in self.tracks:
            track.pad_to_multiple(factor)

    def pad_to_same(self):
        """Pad shorter pianorolls with zeros at the end along the time axis to
        make the resulting pianoroll lengths the same as the maximum pianoroll
        length among all the tracks."""
        max_length = self.get_max_length()
        for track in self.tracks:
            if track.pianoroll.shape[0] < max_length:
                track.pad(max_length - track.pianoroll.shape[0])

    def parse_midi(self, filename, **kwargs):
        """
        Parse a MIDI file.

        Parameters
        ----------
        filename : str
            The name of the MIDI file to be parsed.
        **kwargs:
            See :meth:`pypianoroll.Multitrack.parse_pretty_midi` for full
            documentation.

        """
        pm = pretty_midi.PrettyMIDI(filename)
        self.parse_pretty_midi(pm, **kwargs)

    def parse_pretty_midi(self, pm, mode='max', algorithm='normal',
                          binarized=False, skip_empty_tracks=True,
                          collect_onsets_only=False, threshold=0,
                          first_beat_time=None):
        """
        Parse a :class:`pretty_midi.PrettyMIDI` object. The data type of the
        resulting pianorolls is automatically determined (int if 'mode' is
        'sum', np.uint8 if `mode` is 'max' and `binarized` is False, bool if
        `mode` is 'max' and `binarized` is True).

        Parameters
        ----------
        pm : `pretty_midi.PrettyMIDI` object
            A :class:`pretty_midi.PrettyMIDI` object to be parsed.
        mode : {'max', 'sum'}
            A string that indicates the merging strategy to apply to duplicate
            notes. Default to 'max'.
        algorithm : {'normal', 'strict', 'custom'}
            A string that indicates the method used to get the location of the
            first beat. Notes before it will be dropped unless an incomplete
            beat before it is found (see Notes for more information). Defaults
            to 'normal'.

            - The 'normal' algorithm estimates the location of the first beat by
              :meth:`pretty_midi.PrettyMIDI.estimate_beat_start`.
            - The 'strict' algorithm sets the first beat at the event time of
              the first time signature change. Raise a ValueError if no time
              signature change event is found.
            - The 'custom' algorithm takes argument `first_beat_time` as the
              location of the first beat.

        binarized : bool
            True to binarize the parsed pianorolls before merging duplicate
            notes. False to use the original parsed pianorolls. Defaults to
            False.
        skip_empty_tracks : bool
            True to remove tracks with empty pianorolls and compress the pitch
            range of the parsed pianorolls. False to retain the empty tracks
            and use the original parsed pianorolls. Deafault to True.
        collect_onsets_only : bool
            True to collect only the onset of the notes (i.e. note on events) in
            all tracks, where the note off and duration information are dropped.
            False to parse regular pianorolls.
        threshold : int or float
            A threshold used to binarize the parsed pianorolls. Only effective
            when `binarized` is True. Defaults to zero.
        first_beat_time : float
            The location (in sec) of the first beat. Required and only effective
            when using 'custom' algorithm.

        Notes
        -----
        If an incomplete beat before the first beat is found, an additional beat
        will be added before the (estimated) beat starting time. However, notes
        before the (estimated) beat starting time for more than one beat are
        dropped.

        """
        if mode not in ('max', 'sum'):
            raise ValueError("`mode` must be one of {'max', 'sum'}.")
        if algorithm not in ('strict', 'normal', 'custom'):
            raise ValueError("`algorithm` must be one of {'normal', 'strict', "
                             " 'custom'}.")
        if algorithm == 'custom':
            if not isinstance(first_beat_time, (int, float)):
                raise TypeError("`first_beat_time` must be int or float when "
                                "using 'custom' algorithm.")
            if first_beat_time < 0.0:
                raise ValueError("`first_beat_time` must be a positive number "
                                 "when using 'custom' algorithm.")

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
                                 "algorithm.")
            pm.time_signature_changes.sort(key=lambda x: x.time)
            first_beat_time = pm.time_signature_changes[0].time

        # get tempo change event times and contents
        tc_times, tempi = pm.get_tempo_changes()
        arg_sorted = np.argsort(tc_times)
        tc_times = tc_times[arg_sorted]
        tempi = tempi[arg_sorted]

        beat_times = pm.get_beats(first_beat_time)
        if not len(beat_times):
            raise ValueError("Cannot get beat timings to quantize pianoroll.")
        beat_times.sort()

        n_beats = len(beat_times)
        n_time_steps = self.beat_resolution * n_beats

        # Parse downbeat array
        if not pm.time_signature_changes:
            self.downbeat = None
        else:
            self.downbeat = np.zeros((n_time_steps,), bool)
            self.downbeat[0] = True
            start = 0
            end = start
            for idx, tsc in enumerate(pm.time_signature_changes[:-1]):
                end += np.searchsorted(beat_times[end:],
                                       pm.time_signature_changes[idx+1].time)
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

        # Parse pianoroll
        self.tracks = []
        for instrument in pm.instruments:
            if binarized:
                pianoroll = np.zeros((n_time_steps, 128), bool)
            elif mode == 'max':
                pianoroll = np.zeros((n_time_steps, 128), np.uint8)
            else:
                pianoroll = np.zeros((n_time_steps, 128), int)

            pitches = np.array([note.pitch for note in instrument.notes
                                if note.end > first_beat_time])
            note_on_times = np.array([note.start for note in instrument.notes
                                      if note.end > first_beat_time])
            beat_indices = np.searchsorted(beat_times, note_on_times) - 1
            remained = note_on_times - beat_times[beat_indices]
            ratios = remained / (beat_times_one_more[beat_indices + 1]
                                 - beat_times[beat_indices])
            rounded = np.round((beat_indices + ratios) * self.beat_resolution)
            note_ons = rounded.astype(int)

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
                beat_indices = np.searchsorted(beat_times, note_off_times) - 1
                remained = note_off_times - beat_times[beat_indices]
                ratios = remained / (beat_times_one_more[beat_indices + 1]
                                     - beat_times[beat_indices])
                note_offs = ((beat_indices + ratios)
                             * self.beat_resolution).astype(int)

                for idx, start in enumerate(note_ons):
                    end = note_offs[idx]
                    velocity = instrument.notes[idx].velocity

                    if velocity < 1:
                        continue
                    if binarized and velocity <= threshold:
                        continue

                    if start > 0 and start < n_time_steps:
                        if pianoroll[start - 1, pitches[idx]]:
                            pianoroll[start - 1, pitches[idx]] = 0
                    if end < n_time_steps - 1:
                        if pianoroll[end, pitches[idx]]:
                            end -= 1

                    if binarized:
                        if mode == 'sum':
                            pianoroll[start:end, pitches[idx]] += 1
                        elif mode == 'max':
                            pianoroll[start:end, pitches[idx]] = True
                    elif mode == 'sum':
                        pianoroll[start:end, pitches[idx]] += velocity
                    elif mode == 'max':
                        maximum = np.maximum(
                            pianoroll[start:end, pitches[idx]], velocity)
                        pianoroll[start:end, pitches[idx]] = maximum

            if skip_empty_tracks and not np.any(pianoroll):
                continue

            track = Track(pianoroll, int(instrument.program),
                          instrument.is_drum, instrument.name)
            self.tracks.append(track)

        self.check_validity()

    def plot(self, **kwargs):
        """Plot the pianorolls or save a plot of them. See
        :func:`pypianoroll.plot.plot_multitrack` for full documentation."""
        return plot_multitrack(self, **kwargs)

    def remove_empty_tracks(self):
        """Remove tracks with empty pianorolls."""
        self.remove_tracks(self.get_empty_tracks())

    def remove_tracks(self, track_indices):
        """
        Remove tracks specified by `track_indices`.

        Parameters
        ----------
        track_indices : list
            The indices of the tracks to be removed.

        """
        if isinstance(track_indices, int):
            track_indices = [track_indices]
        self.tracks = [track for idx, track in enumerate(self.tracks)
                       if idx not in track_indices]

    def save(self, filename, compressed=True):
        """
        Save the multitrack pianoroll to a (compressed) npz file, which can be
        later loaded by :meth:`pypianoroll.Multitrack.load`.

        Notes
        -----
        To reduce the file size, the pianorolls are first converted to instances
        of scipy.sparse.csc_matrix, whose component arrays are then collected
        and saved to a npz file.

        Parameters
        ----------
        filename : str
            The name of the npz file to which the mulitrack pianoroll is saved.
        compressed : bool
            True to save to a compressed npz file. False to save to an
            uncompressed npz file. Defaults to True.

        """
        def update_sparse(target_dict, sparse_matrix, name):
            """Turn `sparse_matrix` into a scipy.sparse.csc_matrix and update
            its component arrays to the `target_dict` with key as `name`
            suffixed with its component type string."""
            csc = csc_matrix(sparse_matrix)
            target_dict[name+'_csc_data'] = csc.data
            target_dict[name+'_csc_indices'] = csc.indices
            target_dict[name+'_csc_indptr'] = csc.indptr
            target_dict[name+'_csc_shape'] = csc.shape

        self.check_validity()
        array_dict = {'tempo': self.tempo}
        info_dict = {'beat_resolution': self.beat_resolution,
                     'name': self.name}

        if self.downbeat is not None:
            array_dict['downbeat'] = self.downbeat

        for idx, track in enumerate(self.tracks):
            update_sparse(array_dict, track.pianoroll,
                          'pianoroll_{}'.format(idx))
            info_dict[str(idx)] = {'program': track.program,
                                   'is_drum': track.is_drum,
                                   'name': track.name}

        if not filename.endswith('.npz'):
            filename += '.npz'
        if compressed:
            np.savez_compressed(filename, **array_dict)
        else:
            np.savez(filename, **array_dict)

        compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(filename, 'a') as zip_file:
            zip_file.writestr('info.json', json.dumps(info_dict), compression)

    def to_pretty_midi(self, constant_tempo=None, constant_velocity=100):
        """
        Convert to a :class:`pretty_midi.PrettyMIDI` instance.

        Notes
        -----
        - Only constant tempo is supported by now.
        - The velocities of the converted pianorolls are clipped to [0, 127],
          i.e. values below 0 and values beyond 127 are replaced by 127 and 0,
          respectively.
        - Adjacent nonzero values of the same pitch will be considered a single
          note with their mean as its velocity.

        Parameters
        ----------
        constant_tempo : int
            The constant tempo value of the output object. Defaults to use the
            first element of `tempo`.
        constant_velocity : int
            The constant velocity to be assigned to binarized tracks. Defaults
            to 100.

        Returns
        -------
        pm : `pretty_midi.PrettyMIDI` object
            The converted :class:`pretty_midi.PrettyMIDI` instance.

        """
        self.check_validity()
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo[0])

        # TODO: Add downbeat support -> time signature change events
        # TODO: Add tempo support -> tempo change events
        if constant_tempo is None:
            constant_tempo = self.tempo[0]
        time_step_size = 60. / constant_tempo / self.beat_resolution

        for track in self.tracks:
            instrument = pretty_midi.Instrument(
                program=track.program, is_drum=track.is_drum, name=track.name)
            copied = track.copy()
            if copied.is_binarized():
                copied.assign_constant(constant_velocity)
            copied.clip()
            clipped = copied.pianoroll.astype(np.uint8)
            binarized = (clipped > 0)
            padded = np.pad(binarized, ((1, 1), (0, 0)), 'constant')
            diff = np.diff(padded.astype(np.int8), axis=0)

            positives = np.nonzero((diff > 0).T)
            pitches = positives[0]
            note_ons = positives[1]
            note_on_times = time_step_size * note_ons
            note_offs = np.nonzero((diff < 0).T)[1]
            note_off_times = time_step_size * note_offs

            for idx, pitch in enumerate(pitches):
                velocity = np.mean(clipped[note_ons[idx]:note_offs[idx], pitch])
                note = pretty_midi.Note(
                    velocity=int(velocity), pitch=pitch,
                    start=note_on_times[idx], end=note_off_times[idx])
                instrument.notes.append(note)

            instrument.notes.sort(key=lambda x: x.start)
            pm.instruments.append(instrument)

        return pm

    def transpose(self, semitone):
        """
        Transpose the pianorolls of all tracks by a number of semitones, where
        positive values are for higher key, while negative values are for lower
        key. The drum tracks are ignored.

        Parameters
        ----------
        semitone : int
            The number of semitones to transpose the pianorolls.

        """
        for track in self.tracks:
            if not track.is_drum:
                track.transpose(semitone)

    def trim_trailing_silence(self):
        """Trim the trailing silences of the pianorolls of all tracks. Trailing
        silences are considered globally."""
        active_length = self.get_active_length()
        for track in self.tracks:
            track.pianoroll = track.pianoroll[:active_length]

    def write(self, filename):
        """
        Write the multitrack pianoroll to a MIDI file.

        Parameters
        ----------
        filename : str
            The name of the MIDI file to which the multitrack pianoroll is
            written.

        """
        if not filename.endswith(('.mid', '.midi', '.MID', '.MIDI')):
            filename = filename + '.mid'
        pm = self.to_pretty_midi()
        pm.write(filename)

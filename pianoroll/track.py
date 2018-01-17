"""
Class for single-track piano-rolls with metadata.
"""
from copy import deepcopy
import numpy as np

class Track(object):
    """
    A single-track piano-roll container

    Attributes
    ----------
    pianoroll : np.ndarray, shape=(num_time_step, num_pitch)
        Piano-roll matrix. First dimension represents time. Second dimension
        represents pitch. The lowest pitch is given by ``lowest_pitch``.
    program: int
        Program number according to General MIDI specification. Available
        values are 0 to 127. Default to 0 (Acoustic Grand Piano).
    is_drum : bool
        Drum indicator. True for drums. False for other instruments.
    name : str
        Name of the track.
    lowest_pitch : int
        Indicate the lowest pitch in the piano-roll.
    """

    def __init__(self, pianoroll=None, program=0, is_drum=False, name='unknown',
                 lowest_pitch=0):
        """
        Initialize by assigning attributes

        Parameters
        ----------
        pianoroll : np.ndarray, shape=(num_time_step, num_pitch)
            Piano-roll matrix. First dimension represents time. Second dimension
            represents pitch. The lowest pitch is given by ``lowest_pitch``.
            Available datatypes are bool, int, float.
        program: int
            Program number according to General MIDI specification [1].
            Available values are 0 to 127. Default to 0 (Acoustic Grand Piano).
        is_drum : bool
            Drum indicator. True for drums. False for other instruments. Default
            to False.
        name : str
            Name of the track. Default to 'unknown'.
        lowest_pitch : int
            Indicate the lowest pitch of the piano-roll. Default to zero.

        References
        ----------
        [1] https://www.midi.org/specifications/item/gm-level-1-sound-set
        """
        # initialize attributes
        if pianoroll is None:
            self.pianoroll = np.zeros((0, 0), bool)
        else:
            self.pianoroll = pianoroll
        self.program = program
        self.is_drum = is_drum
        self.lowest_pitch = lowest_pitch
        self.name = name

        # check validity
        self.check_validity()

    def binarize(self, threshold=0):
        """
        Binarize the piano-roll. Do nothing if the piano-roll is already
        binarized

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano-rolls. Default to zero.
        """
        if not self.is_binarized():
            self.pianoroll = (self.pianoroll > threshold)

    def check_validity(self):
        """"Raise error if any invalid attribute found"""
        # pianoroll
        if not isinstance(self.pianoroll, np.ndarray):
            raise TypeError("`pianoroll` must be of np.ndarray type")
        if not (np.issubdtype(self.pianoroll.dtype, np.bool)
                or np.issubdtype(self.pianoroll.dtype, np.int)
                or np.issubdtype(self.pianoroll.dtype, np.float)):
            raise TypeError("Data type of `pianoroll` must be one of bool, int "
                            "and float.")
        if isinstance(self.pianoroll, np.matrix):
            self.pianoroll = np.asarray(self.pianoroll)
        elif self.pianoroll.ndim != 2:
            raise ValueError("`pianoroll` must be a 2D numpy array")
        # program
        if not isinstance(self.program, int):
            raise TypeError("`program` must be of int type")
        if self.program < 0 or self.program > 127:
            raise ValueError("`program` must be in 0 to 127")
        # is_drum
        if not isinstance(self.is_drum, bool):
            raise TypeError("`is_drum` must be of boolean type")
        # lowest_pitch
        if not isinstance(self.lowest_pitch, int):
            raise TypeError("`lowest_pitch` must be of int type")
        if self.lowest_pitch < 0:
            raise ValueError("`lowest_pitch` must be a non-negative integer")
        # name
        if not isinstance(self.name, str):
            raise TypeError("`name` must be of str type")

    def clip(self, lower=0, upper=128):
        """
        Clip the piano-roll by an lower bound and an upper bound specified by
        `lower` and `upper`, respectively

        Parameters
        ----------
        lower : int or float
            The lower bound to clip the piano-roll. Default to 0.
        upper : int or float
            The upper bound to clip the piano-roll. Default to 128.
        """
        np.clip(self.pianoroll, lower, upper, self.pianoroll)

    def compress_pitch_range(self):
        """Compress the piano-roll to active pitch range"""
        lowest, highest = self.get_pitch_range(True)
        self.pianoroll = self.pianoroll[:, lowest:highest]
        self.lowest_pitch += lowest

    def copy(self):
        """
        Return a copy of the object

        Returns
        -------
        copied : `pianoroll.Track` object
            A copy of the object.
        """
        copied = deepcopy(self)
        return copied

    def get_pianoroll(self, binarized=False, threshold=0):
        """
        Return a (binarized) copy of the piano-roll.

        Notes
        -----
        Ignore ``threshold`` if the piano-roll is already binarized

        Parameters
        ----------
        threshold : int or float
            Threshold to binarize the piano-rolls. Default to zero.

        Returns
        -------
        copied :
            A (binarized) copy of the piano-roll
        lowest : int
            Indicate the lowest pitch in the merged piano-roll.
        """
        if binarized and not self.is_binarized():
            copied = (self.pianoroll > threshold)
        else:
            copied = np.copy(self.pianoroll)
        lowest = self.lowest_pitch
        return copied, lowest

    def get_length(self):
        """
        Return the length of the piano-roll without trailing silence (in time
        step)

        Returns
        -------
        length : int
            Length of the piano-roll without trailing silence (in time step).
        """
        non_zero_steps = np.any((self.pianoroll > 0), axis=1)
        inv_last_non_zero_step = np.argmax(np.flip(non_zero_steps, axis=0))
        length = self.pianoroll.shape[0] - inv_last_non_zero_step - 1
        return length

    def get_expanded_pianoroll(self, lowest=0, highest=127):
        """
        Return an copy of the piano-roll which is expanded to a pitch range
        specified by `lowest` and `highest`

        Parameters
        ----------
        lowest : int or float
            The lowest pitch of the expanded piano-roll.
        highest : int or float
            The highest pitch of the expanded piano-roll.

        Returns
        -------
        expanded : np.ndarray, shape=(num_time_step, highest - lowest + 1)
            An expanded copy of the piano-roll
        """
        pianoroll = np.copy(self.pianoroll)

        pitch_range = highest - lowest + 1

        if (self.lowest_pitch == lowest
                and self.pianoroll.shape[1] == pitch_range):
            return pianoroll

        if self.lowest_pitch > lowest:
            to_pad = self.lowest_pitch - lowest
            expanded = np.pad(pianoroll, ((0, 0), (to_pad, 0)), 'constant')
        elif self.lowest_pitch < lowest:
            expanded = pianoroll[:, (lowest - self.lowest_pitch):]

        if expanded.shape[1] < pitch_range:
            to_pad = pitch_range - expanded.shape[1]
            expanded = np.pad(pianoroll, ((0, 0), (0, to_pad)), 'constant')
        elif expanded.shape[1] > pitch_range:
            expanded = expanded[:, :pitch_range]

        return expanded

    def get_pitch_range(self, relative=False):
        """
        Return the pitch range in tuple (lowest, highest)

        Parameters
        ----------
        relative : bool
            True to return the relative pitch range , i.e. the corresponding
            indices in the pitch axis of the piano-roll. False to return the
            absolute pitch range. Default to False.

        Returns
        -------
        lowest : int
            Indicate the lowest pitch in the piano-roll.
        highest : int
            Indicate the highest pitch in the piano-roll.
        """
        lowest = 0
        while not np.any(self.pianoroll[:, lowest]):
            lowest += 1

        highest = self.pianoroll.shape[1] - 1
        while not np.any(self.pianoroll[:, highest]):
            highest -= 1

        if not relative:
            lowest = self.lowest_pitch + lowest
            highest = self.lowest_pitch + highest

        return lowest, highest

    def is_binarized(self):
        """
        Return True if the piano-roll is already binarized. Otherwise, return
        False

        Returns
        -------
        is_binarized : bool
            True if the piano-roll is already binarized; otherwise, False.
        """
        is_binarized = (self.pianoroll.dtype == bool)
        return is_binarized

    def transpose(self, semitone):
        """
        Transpose the piano-roll by ``semitones`` semitones

        Parameters
        ----------
        semitone : int
            Number of semitones transpose the piano-roll.
        """
        self.lowest_pitch += semitone

    def trim_trailing_silence(self):
        """Trim the trailing silence of the piano-roll"""
        length = self.get_length()
        self.pianoroll = self.pianoroll[:length]

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
        represents pitch. The lowest pitch is given by `lowest`.
    program: int
        Program number according to General MIDI specification. Available
        values are 0 to 127. Default to 0 (Acoustic Grand Piano).
    is_drum : bool
        Drum indicator. True for drums. False for other instruments.
    name : str
        Name of the track.
    lowest : int
        Indicate the lowest pitch in the piano-roll.
    """

    def __init__(self, pianoroll=None, program=0, is_drum=False, name='unknown',
                 lowest=0):
        """
        Initialize by assigning attributes

        Parameters
        ----------
        pianoroll : np.ndarray, shape=(num_time_step, num_pitch)
            Piano-roll matrix. First dimension represents time. Second dimension
            represents pitch. The lowest pitch is given by `lowest`. Available
            datatypes are bool, int, float.
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
        if pianoroll is None:
            self.pianoroll = np.zeros((0, 0), bool)
        else:
            self.pianoroll = pianoroll
        self.program = program
        self.is_drum = is_drum
        self.name = name
        self.lowest = lowest

        self.check_validity()

    def __repr__(self):
        return ("Track(program={}, is_drum={}, name={}, lowest={}, "
                "pianoroll=\n{}".format(self.program, self.is_drum, self.name,
                                        self.lowest, self.pianoroll.__str__))

    def __str__(self):
        return ("program : {},\nis_drum : {},\nname : {},\nlowest : {},\n"
                "pianoroll :\n{}".format(self.program, self.is_drum, self.name,
                                         self.lowest, self.pianoroll.__str__))

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
        if self.pianoroll.ndim != 2:
            raise ValueError("`pianoroll` must be a 2D numpy array")
        # program
        if not isinstance(self.program, int):
            raise TypeError("`program` must be of int type")
        if self.program < 0 or self.program > 127:
            raise ValueError("`program` must be in 0 to 127")
        # is_drum
        if not isinstance(self.is_drum, bool):
            raise TypeError("`is_drum` must be of boolean type")
        # lowest
        if not isinstance(self.lowest, int):
            raise TypeError("`lowest` must be of int type")
        if self.lowest < 0:
            raise ValueError("`lowest` must be a non-negative integer")
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

    def compress(self):
        """Compress the piano-roll to active pitch range"""
        lowest, highest = self.get_pitch_range(True)
        self.pianoroll = self.pianoroll[:, lowest:highest]
        self.lowest += lowest

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

    def expand(self, lowest=0, highest=127):
        """
        Expand the piano-roll to a pitch range specified by `lowest` and
        `highest`

        Parameters
        ----------
        lowest : int or float
            The lowest pitch of the expanded piano-roll.
        highest : int or float
            The highest pitch of the expanded piano-roll.
        """
        if self.lowest > lowest:
            to_pad = self.lowest - lowest
            self.pianoroll = np.pad(self.pianoroll, ((0, 0), (to_pad, 0)),
                                    'constant')
        elif self.lowest < lowest:
            self.pianoroll = self.pianoroll[:, (lowest - self.lowest):]

        self.lowest = lowest

        pitch_axis_length = highest - lowest + 1
        if self.pianoroll.shape[1] < pitch_axis_length:
            to_pad = pitch_axis_length - self.pianoroll.shape[1]
            self.pianoroll = np.pad(self.pianoroll, ((0, 0), (0, to_pad)),
                                    'constant')
        elif self.pianoroll.shape[1] > pitch_axis_length:
            self.pianoroll = self.pianoroll[:, :pitch_axis_length]

    def get_pianoroll(self):
        """
        Return a copy of the piano-roll

        Returns
        -------
        copied :
            A copy of the piano-roll.
        lowest : int
            Indicate the lowest pitch in the piano-roll.
        """
        copied = np.copy(self.pianoroll)
        return copied, self.lowest

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
            lowest = self.lowest + lowest
            highest = self.lowest + highest

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
        Transpose the piano-roll by `semitones` semitones

        Parameters
        ----------
        semitone : int
            Number of semitones transpose the piano-roll.
        """
        self.lowest += semitone

    def trim_trailing_silence(self):
        """Trim the trailing silence of the piano-roll"""
        length = self.get_length()
        self.pianoroll = self.pianoroll[:length]

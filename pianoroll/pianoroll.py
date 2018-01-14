"""
Functions to manipulate multi-track and single-track piano-rolls with
metadata.

Only work for :class:`pianoroll.MultiTrack` and :class:`pianoroll.Track`
objects.
"""

from copy import deepcopy

def binarize(p, threshold):
    """
    Return a copy of the object with binarized piano-roll(s)

    Parameters
    ----------
    threshold : int
        Threshold to binarize the piano-roll(s). Default to zero.
    """
    copied = deepcopy(p)
    copied.binarize(threshold=threshold)
    return copied

def compress_pitch_range(p):
    """
    Return a copy of the object with piano-roll(s) compressed to active pitch
    range(s)
    """
    copied = deepcopy(p)
    copied.compress_pitch_range()
    return copied

def copy(p):
    """Return a copy of the object"""
    copied = deepcopy(p)
    return copied

def transpose(p, semitone):
    """
    Return a copy of the object with piano-roll(s) transposed by ``semitones``
    semitones

    Parameters
    ----------
    semitone : int
        Number of semitones to transpose the piano-roll(s).
    """
    copied = deepcopy(p)
    copied.lowest_pitch += semitone
    return copied

def trim_trailing_silence(p):
    """
    Return a copy of the object with trimmed trailing silence of the
    piano-roll(s)
    """
    copied = deepcopy(p)
    length = p.get_length()
    copied.pianoroll = copied.pianoroll[:length]
    return copied

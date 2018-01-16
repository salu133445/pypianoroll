"""
Functions to manipulate multi-track and single-track piano-rolls with
metadata.

Only work for :class:`pianoroll.MultiTrack` and :class:`pianoroll.Track`
objects.
"""
from copy import deepcopy
import numpy as np

def is_pianoroll(arr):
    """
    Return True if the array is a valid piano-roll matrix. Otherwise, return
    False.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("`arr` must be of np.ndarray type")
    if not (np.issubdtype(arr.dtype, np.bool)
            or np.issubdtype(arr.dtype, np.int)
            or np.issubdtype(arr.dtype, np.float)):
        return False
    if arr.ndim != 2:
        return False
    return True

def is_standard_pianoroll(arr):
    """
    Return True if the array is a standard piano-roll matrix that has a
    pitch range under 128. Otherwise, return False.
    """
    if not is_pianoroll(arr):
        return False
    if arr.shape[2] > 128:
        return False
    return True

def clip(obj, upper=128):
    """
    Return a copy of the object with piano-roll(s) clipped by an upper bound
    specified by `upper`

    Parameters
    ----------
    upper : int
        The upper bound to clip the input piano-roll. Default to 128.
    """
    copied = deepcopy(obj)
    copied.clip(upper)
    return copied

def binarize(obj, threshold=0.0):
    """
    Return a copy of the object with binarized piano-roll(s)

    Parameters
    ----------
    threshold : int
        Threshold to binarize the piano-roll(s). Default to zero.
    """
    copied = deepcopy(obj)
    copied.binarize(threshold)
    return copied

def compress_pitch_range(obj):
    """
    Return a copy of the object with piano-roll(s) compressed to active
    pitch range(s)
    """
    copied = deepcopy(obj)
    copied.compress_pitch_range()
    return copied

def copy(obj):
    """Return a copy of the object"""
    copied = deepcopy(obj)
    return copied

def transpose(obj, semitone):
    """
    Return a copy of the object with piano-roll(s) transposed by
    ``semitones`` semitones

    Parameters
    ----------
    semitone : int
        Number of semitones to transpose the piano-roll(s).
    """
    copied = deepcopy(obj)
    copied.lowest_pitch += semitone
    return copied

def trim_trailing_silence(obj):
    """
    Return a copy of the object with trimmed trailing silence of the
    piano-roll(s)
    """
    copied = deepcopy(obj)
    length = copied.get_length()
    copied.pianoroll = copied.pianoroll[:length]
    return copied

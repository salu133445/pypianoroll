"""
Functions to manipulate multi-track and single-track piano-rolls with
metadata.

Only :class:`pypianoroll.Multitrack` and :class:`pypianoroll.Track`
objects are supported for most functions.
"""
from copy import deepcopy
import numpy as np
from .track import Track
from .multitrack import Multitrack

def _check_supported(obj):
    """
    Raise TypeError if the object is not a :class:`pypianoroll.Multitrack`
    or :class:`pypianoroll.Track` object. Otherwise, pass.
    """
    if not (isinstance(obj, Multitrack) or isinstance(obj, Track)):
        raise TypeError("Only `pypianoroll.Multitrack` and `pypianoroll.Track` "
                        "are supported")

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

def binarize(obj, threshold=0):
    """
    Return a copy of the object with binarized piano-roll(s)

    Parameters
    ----------
    threshold : int or float
        Threshold to binarize the piano-roll(s). Default to zero.
    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.binarize(threshold)
    return copied

def clip(obj, lower=0, upper=128):
    """
    Return a copy of the object with piano-roll(s) clipped by a lower bound
    and an upper bound specified by `lower` and `upper`, respectively

    Parameters
    ----------
    lower : int or float
        The lower bound to clip the piano-roll. Default to 0.
    upper : int or float
        The upper bound to clip the piano-roll. Default to 128.
    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.clip(lower, upper)
    return copied

def compress_to_active(obj):
    """
    Return a copy of the object with piano-roll(s) compressed to active
    pitch range(s)
    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.compress_to_active()
    return copied

def copy(obj):
    """Return a copy of the object"""
    _check_supported(obj)
    copied = deepcopy(obj)
    return copied

def expand(obj, lowest=0, highest=127):
    """
    Return a copy of the object with piano-roll(s) expanded or compressed to
    a pitch range specified by `lowest` and `highest`

    Parameters
    ----------
    lowest : int or float
        The lowest pitch of the expanded piano-roll(s).
    highest : int or float
        The highest pitch of the expanded piano-roll(s).
    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.expand(lowest, highest)
    return copied

def plot(obj, **kwargs):
    """
    Plot the object. See :func:`pypianoroll.Multitrack.plot` and
    :func:`pypianoroll.Track.plot` for full documentation.
    """
    _check_supported(obj)
    return obj.plot(**kwargs)

def transpose(obj, semitone):
    """
    Return a copy of the object with piano-roll(s) transposed by
    ``semitones`` semitones

    Parameters
    ----------
    semitone : int
        Number of semitones to transpose the piano-roll(s).
    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.lowest_pitch += semitone
    return copied

def trim_trailing_silence(obj):
    """
    Return a copy of the object with trimmed trailing silence of the
    piano-roll(s)
    """
    _check_supported(obj)
    copied = deepcopy(obj)
    length = copied.get_length()
    copied.pianoroll = copied.pianoroll[:length]
    return copied

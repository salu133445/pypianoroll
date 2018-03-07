"""Utilities for manipulating multi-track and single-track piano-rolls.

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
        raise TypeError("Support only `pypianoroll.Multitrack` and "
                        "`pypianoroll.Track` class objects")

def is_pianoroll(arr):
    """
    Return True if the array is a standard piano-roll matrix. Otherwise,
    return False. Raise TypeError if the input object is not a numpy array.

    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("`arr` must be of np.ndarray type")
    if not (np.issubdtype(arr.dtype, np.bool)
            or np.issubdtype(arr.dtype, np.int)
            or np.issubdtype(arr.dtype, np.float)):
        return False
    if arr.ndim != 2:
        return False
    if arr.shape[1] != 128:
        return False
    return True

def assign_constant(obj, value):
    """
    Assign a constant value to the nonzeros in the piano-roll(s). If a
    piano-roll is not binarized, its data type will be preserved. If a
    piano-roll is binarized, it will be casted to the type of `value`.

    Arguments
    ---------
    value : int or float
        The constant value to be assigned to the nonzeros of the
        piano-roll(s).

    """
    _check_supported(obj)
    obj.assign_constant(value)

def binarize(obj, threshold=0):
    """
    Return a copy of the object with binarized piano-roll(s).

    Parameters
    ----------
    threshold : int or float
        Threshold to binarize the piano-roll(s). Default to zero.

    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.binarize(threshold)
    return copied

def clip(obj, lower=0, upper=127):
    """
    Return a copy of the object with piano-roll(s) clipped by a lower bound
    and an upper bound specified by `lower` and `upper`, respectively.

    Parameters
    ----------
    lower : int or float
        The lower bound to clip the piano-roll. Default to 0.
    upper : int or float
        The upper bound to clip the piano-roll. Default to 127.

    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.clip(lower, upper)
    return copied

def copy(obj):
    """Return a copy of the object."""
    _check_supported(obj)
    copied = deepcopy(obj)
    return copied

def load(filepath):
    """
    Return a :class:`pypianoroll.Multitrack` object loaded from a .npz file.

    Parameters
    ----------
    filepath : str
        The file path to the .npz file.

    """
    if not filepath.endswith('.npz'):
        raise ValueError("Only .npz files are supported")
    multitrack = Multitrack(filepath)
    return multitrack

def pad(obj, pad_length):
    """
    Return a copy of the object with piano-roll padded with zeros at the end
    along the time axis.

    Parameters
    ----------
    pad_length : int
        The length to pad along the time axis with zeros.

    """
    if not isinstance(obj, Track):
        raise TypeError("Support only `pypianoroll.Track` class objects")
    copied = deepcopy(obj)
    copied.pad(pad_length)
    return copied

def pad_to_same(obj):
    """
    Return a copy of the object with shorter piano-rolls padded with zeros
    at the end along the time axis to the length of the piano-roll with the
    maximal length.

    """
    if not isinstance(obj, Multitrack):
        raise TypeError("Support only `pypianoroll.Multitrack` class objects")
    copied = deepcopy(obj)
    copied.pad_to_same()
    return copied

def pad_to_multiple(obj, factor):
    """
    Return a copy of the object with its piano-roll padded with zeros at the
    end along the time axis with the minimal length that make the length of
    the resulting piano-roll a multiple of `factor`.

    Parameters
    ----------
    factor : int
        The value which the length of the resulting piano-roll will be
        a multiple of.

    """
    if not isinstance(obj, Track):
        raise TypeError("Support only `pypianoroll.Track` class objects")
    copied = deepcopy(obj)
    copied.pad_to_multiple(factor)
    return copied

def parse(filepath):
    """
    Return a :class:`pypianoroll.Multitrack` object loaded from a MIDI
    (.mid, .midi, .MID, .MIDI) file.

    Parameters
    ----------
    filepath : str
        The file path to the MIDI file.

    """
    if not filepath.endswith(('.mid', '.midi', '.MID', '.MIDI')):
        raise ValueError("Only MIDI files are supported")
    multitrack = Multitrack(filepath)
    return multitrack

def plot(obj, **kwargs):
    """
    Plot the object. See :func:`pypianoroll.Multitrack.plot` and
    :func:`pypianoroll.Track.plot` for full documentation.

    """
    _check_supported(obj)
    return obj.plot(**kwargs)

def save(filepath, obj, compressed=True):
    """
    Save the object to a .npz file.

    Parameters
    ----------
    filepath : str
        The path to save the file.
    obj: `pypianoroll.Multitrack` objects
        The objecte to be saved.

    """
    if not isinstance(obj, Multitrack):
        raise TypeError("Support only `pypianoroll.Multitrack` class objects")
    obj.save(filepath, compressed)

def transpose(obj, semitone):
    """
    Return a copy of the object with piano-roll(s) transposed by `semitones`
    semitones.

    Parameters
    ----------
    semitone : int
        Number of semitones to transpose the piano-roll(s).

    """
    _check_supported(obj)
    copied = deepcopy(obj)
    copied.transpose(semitone)
    return copied

def trim_trailing_silence(obj):
    """
    Return a copy of the object with trimmed trailing silence of the
    piano-roll(s).

    """
    _check_supported(obj)
    copied = deepcopy(obj)
    length = copied.get_active_length()
    copied.pianoroll = copied.pianoroll[:length]
    return copied

def write(obj, filepath):
    """
    Write the object to a MIDI file.

    Parameters
    ----------
    filepath : str
        The path to write the MIDI file.

    """
    if not isinstance(obj, Multitrack):
        raise TypeError("Support only `pypianoroll.Multitrack` class objects")
    obj.write(filepath)

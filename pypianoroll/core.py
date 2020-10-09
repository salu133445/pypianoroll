"""Utilities for manipulating multitrack pianorolls."""
from typing import Union

from .multitrack import Multitrack
from .track import Track

__all__ = [
    "assign_constant",
    "binarize",
    "clip",
    "pad",
    "pad_to_multiple",
    "pad_to_same",
    "plot",
    "save",
    "transpose",
    "trim_trailing_silence",
    "write",
]


def assign_constant(obj: Union[Multitrack, Track], value: float):
    """Assign a constant value to all nonzeros entries.

    If a piano roll is not binarized, its data type will be preserved.
    If a piano roll is binarized, cast it to the dtype of `value`.

    Arguments
    ---------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
      object
        Object to modify.
    value : int or float
        Value to assign to all the nonzero entries in the piano roll(s).

    """
    return obj.assign_constant(value=value)


def binarize(obj: Union[Multitrack, Track], threshold: float = 0):
    """Binarize the piano roll(s).

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
      object
        Object to binarize.
    threshold : int or float
        Threshold to binarize the piano roll(s). Defaults to zero.

    """
    return obj.binarize(threshold=threshold)


def clip(obj: Union[Multitrack, Track], lower: float = 0, upper: float = 127):
    """Clip the piano roll(s) by a lower bound and an upper bound.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
      object
        Object to clip.
    lower : int or float
        Lower bound to clip the piano roll(s). Defaults to 0.
    upper : int or float
        Upper bound to clip the piano roll(s). Defaults to 127.

    """
    return obj.clip(lower=lower, upper=upper)


def downsample(obj: Multitrack, factor: int):
    """Downsample the piano rolls by the given factor.

    Attribute `resolution` will be updated accordingly as well.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` object
        Object to downsample.
    factor : int
        Ratio of the original resolution to the desired resolution.

    """
    return obj.downsample(factor=factor)


def pad(obj: Union[Multitrack, Track], pad_length: int):
    """Pad the piano roll(s) with zeros at the end along the time axis.

    Notes
    -----
    The lengths of the resulting piano rolls are not guaranteed to be
    the same. See :meth:`pypianoroll.Multitrack.pad_to_same`.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
      object
        Object to pad.
    pad_length : int
        Length to pad along the time axis with zeros.

    """
    return obj.pad(pad_length=pad_length)


def pad_to_multiple(obj: Union[Multitrack, Track], factor: int):
    """Pad the piano rolls along the time axis to a multiple.

    Pad the piano rolls with zeros at the end along the time axis of the
    minimum length that makes the lengths of the resulting piano rolls
    multiples of `factor`.

    Notes
    -----
    The resulting pianoroll lengths are not guaranteed to be the same.
    See :meth:`pypianoroll.Multitrack.pad_to_same`.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
      object
        Object to pad.
    factor : int
        The value which the length of the resulting pianoroll(s) will be
        a multiple of.

    """
    return obj.pad_to_multiple(factor=factor)


def pad_to_same(obj: Union[Multitrack]):
    """Pad piano rolls along the time axis to have the same length.

    Pad shorter piano rolls with zeros at the end along the time axis so
    that the resulting piano rolls have the same length.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` object
        Object to pad.

    """
    return obj.pad_to_same()


def transpose(obj: Union[Multitrack, Track], semitone: int):
    """Transpose the piano roll(s) by a number of semitones.

    Positive values are for a higher key, while negative values are for
    a lower key. Drum tracks are ignored.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
      object
        Object to transpose.
    semitone : int
        Number of semitones to transpose the piano roll(s).

    """
    return obj.transpose(semitone)


def trim_trailing_silence(obj: Union[Multitrack, Track]):
    """Trim the trailing silences of the piano roll(s).

    All the piano rolls will have the same length after the trimming.

    """
    return obj.trim_trailing_silence()


def save(path: str, obj: Multitrack, compressed: bool = True):
    """Save the object to a (compressed) NPZ file.

    This could be later loaded by :func:`pypianoroll.load`.

    Parameters
    ----------
    path : str
        Path to the NPZ file to save.
    obj : :class:`pypianoroll.Multitrack` object
        Object to save.
    compressed : bool
        Whether to save to a compressed NPZ file. Defaults to True.

    Notes
    -----
    To reduce the file size, the piano rolls are first converted to
    instances of :class:`scipy.sparse.csc_matrix`. The component arrays
    are then collected and saved to a npz file.

    """
    obj.save(path, compressed=compressed)


def write(obj: Multitrack, path: str):
    """Write the object to a MIDI file.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` object
        Object to write.
    path : str
        Path to the MIDI file to write.

    """
    obj.write(path)


def plot(obj: Union[Multitrack, Track], **kwargs):
    """Plot the object.

    See :func:`pypianoroll.plot_multitrack` and
    :func:`pypianoroll.plot_track` for full documentation.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
      object
        Object to plot.

    """
    return obj.plot(**kwargs)

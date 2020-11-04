"""Functions for Pypianoroll objects.

Functions
---------

- binarize
- clip
- pad
- pad_to_multiple
- pad_to_same
- plot
- set_nonzeros
- set_resolution
- transpose
- trim


"""
from typing import Optional, TypeVar, Union, overload

from .multitrack import Multitrack
from .track import BinaryTrack, StandardTrack, Track

__all__ = [
    "binarize",
    "clip",
    "pad",
    "pad_to_multiple",
    "pad_to_same",
    "plot",
    "set_nonzeros",
    "set_resolution",
    "transpose",
    "trim",
]

_Multitrack = TypeVar("_Multitrack", bound=Multitrack)
_MultitrackOrTrack = TypeVar("_MultitrackOrTrack", Multitrack, Track)
_StandardTrack = TypeVar("_StandardTrack", Multitrack, StandardTrack)


@overload
def set_nonzeros(obj: Multitrack, value: int) -> Multitrack:
    """Assign a constant value to all nonzeros entries."""


@overload
def set_nonzeros(
    obj: Union[StandardTrack, BinaryTrack], value: int
) -> StandardTrack:
    """Assign a constant value to all nonzeros entries."""


def set_nonzeros(
    obj: Union[Multitrack, StandardTrack, BinaryTrack], value: int
):
    """Assign a constant value to all nonzeros entries.

    Arguments
    ---------
    obj : :class:`pypianoroll.Multitrack`, \
            :class:`pypianoroll.StandardTrack` or \
            :class:`pypianoroll.BinaryTrack`
        Object to modify.
    value : int
        Value to assign.

    """
    return obj.set_nonzeros(value=value)


@overload
def binarize(obj: Multitrack, threshold: int = 0) -> Multitrack:
    """Binarize the piano roll(s)."""


@overload
def binarize(obj: StandardTrack, threshold: int = 0) -> BinaryTrack:
    """Binarize the piano roll(s)."""


def binarize(obj: Union[Multitrack, StandardTrack], threshold: int = 0):
    """Binarize the piano roll(s).

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or \
            :class:`pypianoroll.StandardTrack`
        Object to binarize.
    threshold : int
        Threshold. Defaults to 0.

    """
    return obj.binarize(threshold=threshold)


def clip(
    obj: _StandardTrack, lower: int = 0, upper: int = 127
) -> _StandardTrack:
    """Clip (limit) the the piano roll(s) into [`lower`, `upper`].

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or \
            :class:`pypianoroll.StandardTrack`
        Object to clip.
    lower : int
        Lower bound. Defaults to 0.
    upper : int
        Upper bound. Defaults to 127.

    Returns
    -------
    Object itself.

    """
    return obj.clip(lower=lower, upper=upper)


def set_resolution(
    obj: _Multitrack, resolution: int, rounding: Optional[str] = "round"
) -> _Multitrack:
    """Downsample the piano rolls by a factor.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack`
        Object to downsample.
    resolution : int
        Target resolution.
    rounding : {'round', 'ceil', 'floor'}
        Rounding mode. Defaults to 'round'.

    Returns
    -------
    Object itself.

    """
    return obj.set_resolution(resolution=resolution, rounding=rounding)


def pad(obj: _MultitrackOrTrack, pad_length: int) -> _MultitrackOrTrack:
    """Pad the piano roll(s).

    Notes
    -----
    The lengths of the resulting piano rolls are not guaranteed to be
    the same. See :meth:`pypianoroll.Multitrack.pad_to_same`.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
        Object to pad.
    pad_length : int
            Length to pad along the time axis.

    Returns
    -------
    Object itself.

    See Also
    --------
    :func:`pypianoroll.pad_to_same` : Pad the piano rolls so that they
      have the same length.
    :func:`pypianoroll.pad_to_multiple` : Pad the piano rolls so that
      their lengths are some multiples.

    """
    return obj.pad(pad_length=pad_length)


def pad_to_multiple(
    obj: _MultitrackOrTrack, factor: int
) -> _MultitrackOrTrack:
    """Pad the piano roll(s) so that their lengths are some multiples.

    Pad the piano rolls at the end along the time axis of the
    minimum length that makes the lengths of the resulting piano rolls
    multiples of `factor`.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
        Object to pad.
    factor : int
        The value which the length of the resulting pianoroll(s) will be
        a multiple of.

    Returns
    -------
    Object itself.

    Notes
    -----
    Lengths of the resulting piano rolls are necessarily the same.

    See Also
    --------
    :func:`pypianoroll.pad` : Pad the piano rolls.
    :func:`pypianoroll.pad_to_same` : Pad the piano rolls so that they
      have the same length.

    """
    return obj.pad_to_multiple(factor=factor)


def pad_to_same(obj: _Multitrack) -> _Multitrack:
    """Pad the piano rolls so that they have the same length.

    Pad shorter piano rolls at the end along the time axis so that the
    resulting piano rolls have the same length.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack`
        Object to pad.

    Returns
    -------
    Object itself.

    See Also
    --------
    :func:`pypianoroll.pad` : Pad the piano rolls.
    :func:`pypianoroll.pad_to_multiple` : Pad the piano
      rolls so that their lengths are some multiples.

    """
    return obj.pad_to_same()


def transpose(obj: _MultitrackOrTrack, semitone: int) -> _MultitrackOrTrack:
    """Transpose the piano roll(s) by a number of semitones.

    Positive values are for a higher key, while negative values are for
    a lower key. Drum tracks are ignored.

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
        Object to transpose.
    semitone : int
        Number of semitones to transpose. A positive value raises the
        pitches, while a negative value lowers the pitches.

    Returns
    -------
    Object itself.

    """
    return obj.transpose(semitone=semitone)


def trim(
    obj: _MultitrackOrTrack,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> _MultitrackOrTrack:
    """Trim the trailing silences of the piano roll(s).

    Parameters
    ----------
    obj : :class:`pypianoroll.Multitrack` or :class:`pypianoroll.Track`
        Object to trim.
    start : int, optional
        Start time. Defaults to 0.
    end : int, optional
        End time. Defaults to active length.

    Returns
    -------
    Object itself.

    """
    return obj.trim(start=start, end=end)


def plot(obj: _MultitrackOrTrack, **kwargs) -> _MultitrackOrTrack:
    """Plot the object.

    See :func:`pypianoroll.plot_multitrack` and
    :func:`pypianoroll.plot_track` for full documentation.

    """
    return obj.plot(**kwargs)

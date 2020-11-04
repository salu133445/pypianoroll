"""Objective metrics for piano rolls.

Functions
---------

- drum_in_pattern_rate
- empty_beat_rate
- in_scale_rate
- n_pitch_classes_used
- n_pitches_used
- pitch_range
- pitch_range_tuple
- polyphonic_rate
- qualified_note_rate
- tonal_distance

"""
from math import nan
from typing import Sequence, Tuple

import numpy as np
from numpy import ndarray

__all__ = [
    "drum_in_pattern_rate",
    "empty_beat_rate",
    "in_scale_rate",
    "n_pitch_classes_used",
    "n_pitches_used",
    "pitch_range",
    "pitch_range_tuple",
    "polyphonic_rate",
    "qualified_note_rate",
    "tonal_distance",
]


def _to_chroma(pianoroll: ndarray) -> ndarray:
    """Return the unnormalized chroma features."""
    reshaped = pianoroll[:, :120].reshape(-1, 12, 10)
    reshaped[..., :8] += pianoroll[:, 120:].reshape(-1, 1, 8)
    return np.sum(reshaped, -1)


def empty_beat_rate(pianoroll: ndarray, resolution: int) -> float:
    r"""Return the ratio of empty beats.

    The empty-beat rate is defined as the ratio of the number of empty
    beats (where no note is played) to the total number of beats. Return
    NaN if song length is zero.

    .. math:: empty\_beat\_rate = \frac{\#(empty\_beats)}{\#(beats)}

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.

    Returns
    -------
    float
        Empty-beat rate.

    """
    reshaped = pianoroll.reshape(-1, resolution * pianoroll.shape[1])
    if len(reshaped) < 1:
        return nan
    n_empty_beats = np.count_nonzero(reshaped.any(1))
    return n_empty_beats / len(reshaped)


def n_pitches_used(pianoroll: ndarray) -> int:
    """Return the number of unique pitches used.

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.

    Returns
    -------
    int
        Number of unique pitch classes used.

    See Also
    --------
    :func:`pypianoroll.n_pitch_class_used`: Compute the number of unique
      pitch classes used.

    """
    return np.count_nonzero(np.any(pianoroll, 0))


def n_pitch_classes_used(pianoroll: ndarray) -> int:
    """Return the number of unique pitch classes used.

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.

    Returns
    -------
    int
        Number of unique pitch classes used.

    See Also
    --------
    :func:`pypianoroll.n_pitches_used`: Compute the number of unique
      pitches used.

    """
    return np.count_nonzero(_to_chroma(pianoroll).any(0))


def pitch_range_tuple(pianoroll) -> Tuple[float, float]:
    """Return the pitch range as a tuple `(lowest, highest)`.

    Returns
    -------
    int or nan
        Highest active pitch.
    int or nan
        Lowest active pitch.

    See Also
    --------
    :func:`pypianoroll.pitch_range`: Compute the pitch range.

    """
    nonzero = pianoroll.any(0).nonzero()[0]
    if not nonzero.size:
        return nan, nan
    return nonzero[0], nonzero[-1]


def pitch_range(pianoroll) -> float:
    """Return the pitch range.

    Returns
    -------
    int or nan
        Pitch range (in semitones), i.e., difference between the
        highest and the lowest active pitches.

    See Also
    --------
    :func:`pypianoroll.pitch_range_tuple`: Return the pitch range as a
      tuple.

    """
    lowest, highest = pitch_range_tuple(pianoroll)
    return highest - lowest


def qualified_note_rate(pianoroll: ndarray, threshold: float = 2) -> float:
    r"""Return the ratio of the number of the qualified notes.

    The qualified note rate is defined as the ratio of the number of
    qualified notes (notes longer than `threshold`, in time steps) to
    the total number of notes. Return NaN if no note is found.

    .. math::
        qualified\_note\_rate = \frac{
            \#(notes\_longer\_than\_the\_threshold)
        }{
            \#(notes)
        }

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.
    threshold : int
        Threshold of note length to count into the numerator.

    Returns
    -------
    float
        Qualified note rate.

    References
    ----------
    1. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang,
       "MuseGAN: Multi-track sequential generative adversarial networks
       for symbolic music generation and accompaniment," in Proceedings
       of the 32nd AAAI Conference on Artificial Intelligence (AAAI),
       2018.

    """
    if np.issubdtype(pianoroll.dtype, np.bool_):
        pianoroll = pianoroll.astype(np.uint8)
    padded = np.pad(pianoroll, ((1, 1), (0, 0)), "constant")
    diff = np.diff(padded, axis=0).reshape(-1)
    onsets = (diff > 0).nonzero()[0]
    if len(onsets) < 1:
        return nan
    offsets = (diff < 0).nonzero()[0]
    n_qualified_notes = np.count_nonzero(offsets - onsets >= threshold)
    return n_qualified_notes / len(onsets)


def polyphonic_rate(pianoroll: ndarray, threshold: float = 2) -> float:
    r"""Return the ratio of time steps where multiple pitches are on.

    The polyphony rate is defined as the ratio of the number of time
    steps where multiple pitches are on to the total number of time
    steps. Drum tracks are ignored. Return NaN if song length is zero.
    This metric is used in [1], where it is called *polyphonicity*.

    .. math::
        polyphony\_rate = \frac{
            \#(time\_steps\_where\_multiple\_pitches\_are\_on)
        }{
            \#(time\_steps)
        }

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.
    threshold : int
        Threshold of number of pitches to count into the numerator.

    Returns
    -------
    float
        Polyphony rate.

    References
    ----------
    1. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang,
       "MuseGAN: Multi-track sequential generative adversarial networks
       for symbolic music generation and accompaniment," in Proceedings
       of the 32nd AAAI Conference on Artificial Intelligence (AAAI),
       2018.

    """
    if len(pianoroll) < 1:
        return nan
    n_poly = np.count_nonzero(np.count_nonzero(pianoroll, 1) > threshold)
    return n_poly / len(pianoroll)


def drum_in_pattern_rate(
    pianoroll: ndarray, resolution: int, tolerance: float = 0.1
) -> float:
    r"""Return the ratio of drum notes in a certain drum pattern.

    The drum-in-pattern rate is defined as the ratio of the number of
    notes in a certain scale to the total number of notes. Only drum
    tracks are considered. Return NaN if no drum note is found. This
    metric is used in [1].

    .. math::
        drum\_in\_pattern\_rate = \frac{
            \#(drum\_notes\_in\_pattern)}{\#(drum\_notes)}

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.
    resolution : int
        Time steps per beat.
    tolerance : float
        Tolerance. Defaults to 0.1.

    Returns
    -------
    float
        Drum-in-pattern rate.

    References
    ----------
    1. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang,
       "MuseGAN: Multi-track sequential generative adversarial networks
       for symbolic music generation and accompaniment," in Proceedings
       of the 32nd AAAI Conference on Artificial Intelligence (AAAI),
       2018.

    """
    if resolution not in (4, 6, 8, 9, 12, 16, 18, 24):
        raise ValueError(
            "Unsupported beat resolution. Expect 4, 6, 8 ,9, 12, 16, 18 or 24."
        )

    def _drum_pattern_mask(res, tol):
        """Return a drum pattern mask with the given tolerance."""
        if res == 24:
            drum_pattern_mask = np.tile([1.0, tol, 0.0, 0.0, 0.0, tol], 4)
        elif res == 12:
            drum_pattern_mask = np.tile([1.0, tol, tol], 4)
        elif res == 6:
            drum_pattern_mask = np.tile([1.0, tol, tol], 2)
        elif res == 18:
            drum_pattern_mask = np.tile([1.0, tol, 0.0, 0.0, 0.0, tol], 3)
        elif res == 9:
            drum_pattern_mask = np.tile([1.0, tol, tol], 3)
        elif res == 16:
            drum_pattern_mask = np.tile([1.0, tol, 0.0, tol], 4)
        elif res == 8:
            drum_pattern_mask = np.tile([1.0, tol], 4)
        elif res == 4:
            drum_pattern_mask = np.tile([1.0, tol], 2)
        return drum_pattern_mask

    drum_pattern_mask = _drum_pattern_mask(resolution, tolerance)
    n_in_pattern = np.sum(drum_pattern_mask * np.count_nonzero(pianoroll, 1))
    return n_in_pattern / np.count_nonzero(pianoroll)


def _get_scale(root: int, mode: str) -> ndarray:
    """Return the scale mask for a specific root."""
    if mode == "major":
        a_scale_mask = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], bool)
    else:
        a_scale_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
    return np.roll(a_scale_mask, root)


def in_scale_rate(
    pianoroll: ndarray, root: int = 3, mode: str = "major"
) -> float:
    r"""Return the ratio of pitches in a certain musical scale.

    The pitch-in-scale rate is defined as the ratio of the number of
    notes in a certain scale to the total number of notes. Drum tracks
    are ignored. Return NaN if no note is found. This metric is used in
    [1].

    .. math::
        pitch\_in\_scale\_rate = \frac{\#(notes\_in\_scale)}{\#(notes)}

    Parameters
    ----------
    pianoroll : ndarray
        Piano roll to evaluate.
    root : int
        Root of the scale.
    mode : str, {'major', 'minor'}
        Mode of the scale.

    Returns
    -------
    float
        Pitch-in-scale rate.

    See Also
    --------
    :func:`muspy.scale_consistency`: Compute the largest pitch-in-class
      rate.

    References
    ----------
    1. Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang,
       "MuseGAN: Multi-track sequential generative adversarial networks
       for symbolic music generation and accompaniment," in Proceedings
       of the 32nd AAAI Conference on Artificial Intelligence (AAAI),
       2018.

    """
    chroma = _to_chroma(pianoroll)
    scale_mask = _get_scale(root, mode)
    n_in_scale = np.sum(scale_mask.reshape(-1, 12) * chroma)
    return n_in_scale / np.count_nonzero(pianoroll)


def _get_tonal_matrix(r1, r2, r3) -> ndarray:
    """Return a tonal matrix for computing the tonal distance."""
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7.0 / 6.0) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7.0 / 6.0) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3.0 / 2.0) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3.0 / 2.0) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2.0 / 3.0) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2.0 / 3.0) * np.pi)
    return tonal_matrix


def _to_tonal_space(
    pianoroll: ndarray, resolution: int, tonal_matrix: ndarray
) -> ndarray:
    """Return the tensor in tonal space (chroma normalized per beat)."""
    beat_chroma = _to_chroma(pianoroll).reshape((-1, resolution, 12))
    beat_chroma = beat_chroma / beat_chroma.sum(2, keepdims=True)
    return np.matmul(tonal_matrix, beat_chroma.T).T


def tonal_distance(
    pianoroll_1: ndarray,
    pianoroll_2: ndarray,
    resolution: int,
    radii: Sequence[float] = (1.0, 1.0, 0.5),
) -> float:
    """Return the tonal distance [1] between the two input piano rolls.

    Parameters
    ----------
    pianoroll_1 : ndarray
        First piano roll to evaluate.
    pianoroll_2 : ndarray
        Second piano roll to evaluate.
    resolution : int
        Time steps per beat.
    radii : tuple of float
        Radii of the three tonal circles (see Equation 3 in [1]).

    References
    ----------
    1. Christopher Harte, Mark Sandler, and Martin Gasser, "Detecting
       harmonic change in musical audio," in Proceedings of the 1st ACM
       workshop on Audio and music computing multimedia, 2006.

    """
    assert len(pianoroll_1) == len(
        pianoroll_2
    ), "Input piano rolls must have the same length."

    r1, r2, r3 = radii
    tonal_matrix = _get_tonal_matrix(r1, r2, r3)
    mapped_1 = _to_tonal_space(pianoroll_1, resolution, tonal_matrix)
    mapped_2 = _to_tonal_space(pianoroll_2, resolution, tonal_matrix)
    return np.linalg.norm(mapped_1 - mapped_2)

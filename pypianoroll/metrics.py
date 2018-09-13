"""Metrics for evaluating multitrack and single-track pianorolls for automatic
music generation systems.
"""
import numpy as np

def _validate_pianoroll(pianoroll):
    """Raise an error if the input array is not a standard pianoroll."""
    if not isinstance(pianoroll, np.ndarray):
        raise TypeError("`pianoroll` must be of np.ndarray type.")
    if not (np.issubdtype(pianoroll.dtype, np.bool_)
            or np.issubdtype(pianoroll.dtype, np.number)):
        raise TypeError("The data type of `pianoroll` must be np.bool_ or a "
                        "subdtype of  np.number.")
    if pianoroll.ndim != 2:
        raise ValueError("`pianoroll` must have exactly two dimensions.")
    if pianoroll.shape[1] != 128:
        raise ValueError("The length of the second axis of `pianoroll` must be "
                         "128.")

def _to_chroma(pianoroll):
    """Return the unnormalized chroma features of a pianoroll."""
    _validate_pianoroll(pianoroll)
    reshaped = pianoroll[:, :120].reshape(-1, 12, 10)
    reshaped[..., :8] += pianoroll[:, 120:].reshape(-1, 1, 8)
    return np.sum(reshaped, 1)

def empty_beat_rate(pianoroll, beat_resolution):
    """Return the ratio of empty beats to the total number of beats in a
    pianoroll."""
    _validate_pianoroll(pianoroll)
    reshaped = pianoroll.reshape(-1, beat_resolution * pianoroll.shape[1])
    n_empty_beats = np.count_nonzero(reshaped.any(1))
    return n_empty_beats / len(reshaped)

def n_pitches_used(pianoroll):
    """Return the number of unique pitches used in a pianoroll."""
    _validate_pianoroll(pianoroll)
    return np.count_nonzero(np.any(pianoroll, 0))

def n_pitche_classes_used(pianoroll):
    """Return the number of unique pitch classes used in a pianoroll."""
    _validate_pianoroll(pianoroll)
    chroma = _to_chroma(pianoroll)
    return np.count_nonzero(np.any(chroma, 0))

def qualified_note_rate(pianoroll, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a pianoroll."""
    _validate_pianoroll(pianoroll)
    if np.issubdtype(pianoroll.dtype, np.bool_):
        pianoroll = pianoroll.astype(np.uint8)
    padded = np.pad(pianoroll, ((1, 1), (0, 0)), 'constant')
    diff = np.diff(padded, axis=0).reshape(-1)
    onsets = (diff > 0).nonzero()[0]
    offsets = (diff < 0).nonzero()[0]
    n_qualified_notes = np.count_nonzero(offsets - onsets >= threshold)
    return n_qualified_notes / len(onsets)

def polyphonic_rate(pianoroll, threshold=2):
    """Return the ratio of the number of time steps where the number of pitches
    being played is larger than `threshold` to the total number of time steps
    in a pianoroll."""
    _validate_pianoroll(pianoroll)
    n_poly = np.count_nonzero(np.count_nonzero(pianoroll, 1) > threshold)
    return n_poly / len(pianoroll)

def drum_in_pattern_rate(pianoroll, beat_resolution, tolerance=0.1):
    """Return the ratio of the number of drum notes that lie on the drum
    pattern (i.e., at certain time steps) to the total number of drum notes."""
    if beat_resolution not in (4, 6, 8, 9, 12, 16, 18, 24):
        raise ValueError("Unsupported beat resolution. Only 4, 6, 8 ,9, 12, "
                         "16, 18, 42 are supported.")
    _validate_pianoroll(pianoroll)

    def _drum_pattern_mask(res, tol):
        """Return a drum pattern mask with the given tolerance."""
        if res == 24:
            drum_pattern_mask = np.tile([1., tol, 0., 0., 0., tol], 4)
        elif res == 12:
            drum_pattern_mask = np.tile([1., tol, tol], 4)
        elif res == 6:
            drum_pattern_mask = np.tile([1., tol, tol], 2)
        elif res == 18:
            drum_pattern_mask = np.tile([1., tol, 0., 0., 0., tol], 3)
        elif res == 9:
            drum_pattern_mask = np.tile([1., tol, tol], 3)
        elif res == 16:
            drum_pattern_mask = np.tile([1., tol, 0., tol], 4)
        elif res == 8:
            drum_pattern_mask = np.tile([1., tol], 4)
        elif res == 4:
            drum_pattern_mask = np.tile([1., tol], 2)
        return drum_pattern_mask

    drum_pattern_mask = _drum_pattern_mask(beat_resolution, tolerance)
    n_in_pattern = np.sum(drum_pattern_mask * np.count_nonzero(pianoroll, 1))
    return n_in_pattern / np.count_nonzero(pianoroll)

def in_scale_rate(pianoroll, key=3, kind='major'):
    """Return the ratio of the number of nonzero entries that lie in a specific
    scale to the total number of nonzero entries in a pianoroll. Default to C
    major scale."""
    if not isinstance(key, int):
        raise TypeError("`key` must an integer.")
    if key > 11 or key < 0:
        raise ValueError("`key` must be in an integer in between 0 and 11.")
    if kind not in ('major', 'minor'):
        raise ValueError("`kind` must be one of 'major' or 'minor'.")
    _validate_pianoroll(pianoroll)

    def _scale_mask(key, kind):
        """Return a scale mask for the given key. Default to C major scale."""
        if kind == 'major':
            a_scale_mask = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], bool)
        else:
            a_scale_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
        return np.roll(a_scale_mask, key)

    chroma = _to_chroma(pianoroll)
    scale_mask = _scale_mask(key, kind)
    n_in_scale = np.sum(scale_mask.reshape(-1, 12) * chroma)
    return n_in_scale / np.count_nonzero(pianoroll)

def tonal_distance(pianoroll_1, pianoroll_2, beat_resolution, r1=1.0, r2=1.0,
                   r3=0.5):
    """Return the tonal distance [1] between the two input pianorolls.

    [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting
        harmonic change in musical audio. In Proc. ACM Workshop on Audio and
        Music Computing Multimedia, 2006.

    """
    _validate_pianoroll(pianoroll_1)
    _validate_pianoroll(pianoroll_2)
    assert len(pianoroll_1) == len(pianoroll_2), (
        "Input pianorolls must have the same length.")

    def _tonal_matrix(r1, r2, r3):
        """Return a tonal matrix for computing the tonal distance."""
        tonal_matrix = np.empty((6, 12))
        tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
        tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
        tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
        tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
        tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
        tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
        return tonal_matrix

    def _to_tonal_space(pianoroll, tonal_matrix):
        """Return the tensor in tonal space where chroma features are normalized
        per beat."""
        beat_chroma = _to_chroma(pianoroll).reshape(-1, beat_resolution, 12)
        beat_chroma = beat_chroma / np.sum(beat_chroma, 2, keepdims=True)
        return np.matmul(tonal_matrix, beat_chroma.T).T

    tonal_matrix = _tonal_matrix(r1, r2, r3)
    mapped_1 = _to_tonal_space(pianoroll_1, tonal_matrix)
    mapped_2 = _to_tonal_space(pianoroll_2, tonal_matrix)
    return np.linalg.norm(mapped_1 - mapped_2)

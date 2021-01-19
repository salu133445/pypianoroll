"""Utility functions.

Functions
---------

- decompose_sparse
- reconstruct_sparse
- hmean

"""
from typing import Dict

from numpy import ndarray
from scipy.sparse import csc_matrix

import numpy as np

def decompose_sparse(matrix: ndarray, name: str) -> Dict[str, ndarray]:
    """Decompose a matrix to sparse components.

    Convert a matrix to a :class:`scipy.sparse.csc_matrix` object.
    Return its component arrays as a dictionary with key as `name`
    suffixed with their component types.

    """
    csc = csc_matrix(matrix)
    return {
        name + "_csc_data": csc.data,
        name + "_csc_indices": csc.indices,
        name + "_csc_indptr": csc.indptr,
        name + "_csc_shape": csc.shape,
    }


def reconstruct_sparse(data_dict: Dict[str, ndarray], name: str) -> ndarray:
    """Reconstruct a matrix from a dictionary."""
    sparse_matrix = csc_matrix(
        (
            data_dict[name + "_csc_data"],
            data_dict[name + "_csc_indices"],
            data_dict[name + "_csc_indptr"],
        ),
        shape=data_dict[name + "_csc_shape"],
    )
    return sparse_matrix.toarray()

def hmean(a, axis=0, dtype=None):
    """
    Calculates the harmonic mean along the specified axis.
    That is:  n / (1/x1 + 1/x2 + ... + 1/xn)
    Parameters
    ----------
    a : array_like
        Input array, masked array or object that can be converted to an array.
    axis : int, optional, default axis=0
        Axis along which the harmonic mean is computed.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed. If `dtype` is not specified, it defaults to the
        dtype of `a`, unless `a` has an integer `dtype` with a precision less
        than that of the default platform integer. In that case, the default
        platform integer is used.
    Returns
    -------
    hmean : ndarray
        see `dtype` parameter above
    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.average : Weighted average
    gmean : Geometric mean
    Notes
    -----
    The harmonic mean is computed over a single dimension of the input
    array, axis=0 by default, or all values in the array if axis=None.
    float64 intermediate and return values are used for integer inputs.
    Use masked arrays to ignore any non-finite values in the input or that
    arise in the calculations such as Not a Number and infinity.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=dtype)
    if np.all(a > 0):  # Harmonic mean only defined if greater than zero
        if isinstance(a, np.ma.MaskedArray):
            size = a.count(axis)
        else:
            if axis is None:
                a = a.ravel()
                size = a.shape[0]
            else:
                size = a.shape[axis]
        return size / np.sum(1.0/a, axis=axis, dtype=dtype)
    else:
        raise ValueError("Harmonic mean only defined if all elements greater than zero")
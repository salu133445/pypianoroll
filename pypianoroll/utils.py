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
===========
Save & Load
===========

Pypianoroll supports efficient I/O for :class:`pypianoroll.Multitrack` objects. The piano rolls will be first converted to sparse matrices (specifically, instances of :class:`scipy.sparse.csc_matrix`) and then stored in a NPZ file.


Functions
=========

.. autofunction:: pypianoroll.save
    :noindex:

.. note::
    The saved .npz file is basically a zip archive which contains the following files:

    - component arrays of the piano rolls in compressed sparse column format:

        - ``pianoroll_[index]_csc_data.npy``
        - ``pianoroll_[index]_csc_indices.npy``
        - ``pianoroll_[index]_csc_indptr.npy``

    - ``tempo.npy``: the tempo array
    - ``downbeat.npy``: the down beat array
    - ``info.json``: a JSON file that contains meta data and track information

.. autofunction:: pypianoroll.load
    :noindex:

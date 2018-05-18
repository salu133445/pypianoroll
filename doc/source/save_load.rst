.. _save_load:

Save & Load
===========
Pypianoroll supports efficient utilities for saving and loading
:class:`pypianoroll.Multitrack` objects. The piano-rolls will be first converted
to instances of ``scipy.sparse.csc_matrix`` and then stored in a .npz file.

.. autofunction:: pypianoroll.save

.. note::
    The saved .npz file is basically a zip archive which contains the following
    files:

    - component arrays of the piano-rolls in compressed sparse column format:

        - ``pianoroll_[index]_csc_data.npy``
        - ``pianoroll_[index]_csc_indices.npy``
        - ``pianoroll_[index]_csc_indptr.npy``

    - ``tempo.npy``: the tempo array
    - ``downbeat.npy``: the down beat array
    - ``info.json``: a JSON file that contains meta data and track information

.. autofunction:: pypianoroll.load

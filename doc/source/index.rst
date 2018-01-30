.. include:: ../../README.rst

Documentation
=============

.. toctree::
    :titlesonly:

    getting_started
    save_load
    parse_write
    visualization
    utilities
    track
    multitrack

Why Pypianoroll
===============

Our aim is to provide convenient classes for piano-roll matrix and MIDI-like
track information (program number, track name, drum track indicator).
Since piano-rolls have long been considered not a good way to store music due to
its sparse property, we use scipy sparse matrices for the data I/O.

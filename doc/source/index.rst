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

Lakh Pianoroll Dataset
======================
The `Lakh Pianoroll Dataset<https://salu133445.github.io/musegan/dataset>`_
(LPD) is a collection of 174,154 unique multi-track piano-rolls derived from the
MIDI files in `Lakh MIDI Dataset<http://colinraffel.com/projects/lmd/>`_ (LMD).

The multi-track piano-rolls in LPD are stored using Pypianoroll format.
It's available `here<https://salu133445.github.io/musegan/dataset>`_.
You can play with it if you're interested.

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
    metrics
    track
    multitrack

Why Pypianoroll
===============

Our aim is to provide convenient classes for pianoroll matrix and MIDI-like
track information (program number, track name, drum track indicator).
Pypianoroll is also designed to provide efficient I/O for pianorolls, since
pianorolls have long been considered an inefficient way to store music data due
to the sparse nature.

Lakh Pianoroll Dataset
======================

`Lakh Pianoroll Dataset`_ (LPD) is a new multitrack pianoroll dataset using
Pypianoroll for efficient data I/O and to save space, which is used as the
training dataset in our MuseGAN_ project.

.. _Lakh Pianoroll Dataset: https://salu133445.github.io/musegan/dataset
.. _MuseGAN: https://salu133445.github.io/musegan

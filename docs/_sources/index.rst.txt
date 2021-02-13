===========
Pypianoroll
===========

.. image:: https://img.shields.io/github/workflow/status/salu133445/pypianoroll/Testing
    :target: https://github.com/salu133445/pypianoroll/actions
    :alt: GitHub workflow
.. image:: https://img.shields.io/codecov/c/github/salu133445/pypianoroll
    :target: https://codecov.io/gh/salu133445/pypianoroll
    :alt: Codecov
.. image:: https://img.shields.io/github/license/salu133445/pypianoroll
    :target: https://github.com/salu133445/pypianoroll/blob/main/LICENSE
    :alt: GitHub license
.. image:: https://img.shields.io/github/v/release/salu133445/pypianoroll
    :target: https://github.com/salu133445/pypianoroll/releases
    :alt: GitHub release

Pypianoroll is an open source Python library for working with piano rolls. It provides essential tools for handling multitrack piano rolls, including efficient I/O as well as manipulation, visualization and evaluation tools.


Features
========

- Manipulate multitrack piano rolls intuitively
- Visualize multitrack piano rolls beautifully
- Save and load multitrack piano rolls in a space-efficient format
- Parse MIDI files into multitrack piano rolls
- Write multitrack piano rolls into MIDI files


Why Pypianoroll
===============

Our aim is to provide convenient classes for piano-roll matrix and MIDI-like track information (program number, track name, drum track indicator). Pypianoroll is also designed to provide efficient I/O for piano rolls, since piano rolls have long been considered an inefficient way to store music data due to the sparse nature.


Installation
============

To install Pypianoroll, please run ``pip install pypianoroll``. To build Pypianoroll from source, please download the `source <https://github.com/salu133445/pypianoroll/releases>`_ and run ``python setup.py install``.


Documentation
=============

Documentation is available `here <https://salu133445.github.io/pypianoroll>`_ and as docstrings with the code.


Citing
======

Please cite the following paper if you use Pypianoroll in a published work:

Hao-Wen Dong, Wen-Yi Hsiao, and Yi-Hsuan Yang, "Pypianoroll: Open Source Python Package for Handling Multitrack Pianorolls," in *Late-Breaking Demos of the 19th International Society for Music Information Retrieval Conference (ISMIR)*, 2018.

[`homepage <https://salu133445.github.io/pypianoroll/>`_]
[`paper <https://salu133445.github.io/pypianoroll/pdf/pypianoroll-ismir2018-lbd-paper.pdf>`_]
[`poster <https://salu133445.github.io/pypianoroll/pdf/pypianoroll-ismir2018-lbd-poster.pdf>`_]
[`code <https://github.com/salu133445/pypianoroll>`_]
[`documentation <https://salu133445.github.io/pypianoroll/>`_]


Lakh Pianoroll Dataset
======================

`Lakh Pianoroll Dataset <https://salu133445.github.io/musegan/dataset>`_ (LPD) is a new multitrack piano roll dataset using Pypianoroll for efficient data I/O and to save space, which is used as the training dataset in our `MuseGAN <https://salu133445.github.io/musegan>`_ project.


Contents
========

.. toctree::
    :titlesonly:

    getting_started
    classes
    save_load
    read_write
    visualization
    metrics
    doc

Pypianoroll
===========

A python package for handling multi-track piano-rolls.

.. image:: https://badge.fury.io/py/pypianoroll.svg
   :target: https://badge.fury.io/py/pypianoroll
.. image:: https://travis-ci.org/salu133445/pypianoroll.svg
   :target: https://travis-ci.org/salu133445/pypianoroll
.. image:: https://coveralls.io/repos/github/salu133445/pypianoroll/badge.svg
   :target: https://coveralls.io/github/salu133445/pypianoroll
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/salu133445/musegan/blob/master/LICENSE.txt

Features
--------

- handle piano-rolls of multiple tracks with metadata
- utilities for manipulating piano-rolls
- save to and load from .npz files using efficient sparse matrix format
- parse from and write to MIDI files

Installation
------------

To install Pypianoroll from PYPI:

.. code-block:: bash

    $ pip install pypianoroll

To install Pypianoroll manually (please download the source code from either
PYPI_ or Github_ first):

.. code-block:: bash

    $ python setup.py install

Documentation
-------------

Documentation is provided as docstrings with the code. An online version is
also available here_.

Citing
------

Please cite the following paper if you use Pypianoroll in a published work:

Hao-Wen Dong, Wen-Yi Hsiao, and Yi-Hsuan Yang,
"Pypianoroll: Open Source Python Package for Handling Multitrack Pianorolls,"
in Late-Breaking Demos of the 18th International Society for Music Information
Retrieval Conference (ISMIR), 2018.
[paper_] [poster_]

.. _PYPI: https://pypi.python.org/pypi/pypianoroll
.. _Github: https://github.com/salu133445/pypianoroll
.. _here: https://salu133445.github.io/pypianoroll/
.. _paper: pdf/pypianoroll-ismir2018-lbd-paper.pdf
.. _poster: pdf/pypianoroll-ismir2018-lbd-poster.pdf

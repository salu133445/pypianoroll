============
Read & Write
============

Pypianoroll currently supports reading from and writing to MIDI files.

Functions
=========

.. autofunction:: pypianoroll.read
    :noindex:

.. autofunction:: pypianoroll.write
    :noindex:

.. autofunction:: pypianoroll.from_pretty_midi
    :noindex:

.. autofunction:: pypianoroll.to_pretty_midi
    :noindex:

.. note::
    Writing the tempo array and downbeat array to tempo change and time signature change events are not supported yet.

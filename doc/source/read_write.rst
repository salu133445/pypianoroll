============
Read & Write
============

Pypianoroll currently supports reading from and writing to MIDI files. Supports for MusicXML and ABC format are in the plan.

Note that the MIDI reading and writing functions are based on the Python package pretty_midi. If you use them in a published work, please cite pretty_midi as instructed `here <http://craffel.github.io/pretty-midi/>`_.


Functions
=========

.. autofunction:: pypianoroll.read
    :noindex:

.. autofunction:: pypianoroll.write
    :noindex:

.. note::
    Writing the tempo array and downbeat array to tempo change and time
    signature change events are not supported yet.

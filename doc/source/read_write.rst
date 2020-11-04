============
Read & Write
============

Pypianoroll currently supports reading from and writing to MIDI files. Supports for MusicXML and ABC format are in the plan.

.. note::
    The MIDI reading and writing functions are built on top of pretty_midi. If you use these functions in a published work, please cite pretty_midi as instructed `here <http://craffel.github.io/pretty-midi/>`_.


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

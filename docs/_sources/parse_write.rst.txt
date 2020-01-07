.. _parse_write:

Parse & Write
=============

Pypianoroll currently supports parsing from and writing to MIDI files. Supports
for MusicXML and ABC format are in the plan.

Note that the MIDI parsing and writing functions are based on the Python package
`pretty_midi`. If you use them in a published work, please cite `pretty_midi` as
instructed here_.

Functions
---------

.. autofunction:: pypianoroll.parse
.. autofunction:: pypianoroll.write

.. note::
    Writing the tempo array and down beat array to tempo change and time
    signature change events are not supported yet.

.. _here: http://craffel.github.io/pretty-midi/

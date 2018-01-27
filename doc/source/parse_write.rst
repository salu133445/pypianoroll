.. _parse_write:

Parse & Write
=============
Pypianoroll currently supports parsing from and writing to MIDI files. Supports
for MusicXML and ABC format are in the plan.

Related Functions
-----------------

.. autofunction:: pypianoroll.parse
.. autofunction:: pypianoroll.write

.. note::
    Writing the tempo array and down beat array to tempo change and time
    signature change events are not supported yet.

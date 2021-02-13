===================
Pypianoroll Classes
===================

We provide several classes for working with multitrack piano rolls.

- :class:`pypianoroll.Multitrack`: A multitrack that can store several tracks (:class:`pypianoroll.Track`) along with global information.
- :class:`pypianoroll.Track`: A track that store the piano roll along with track information.
- :class:`pypianoroll.StandardTrack`: A standard track that holds piano roll of type uint8.
- - :class:`pypianoroll.BinaryTrack`: A binary track that holds piano roll of type bool.

Multitrack Class
================

========== =========================== ================================== ==================================
Attribute  Description                 Type                               Default
========== =========================== ================================== ==================================
name       Name of the multitrack      str
resolution Time steps per quarter note int                                ``pypianoroll.DEFAULT_RESOLUTION``
tempo      Tempo at each time step     NumPy array of dtype float
downbeat   Downbeat positions          NumPy array of dtype bool
tracks     Music tracks                list of :class:`pypianoroll.Track` []
========== =========================== ================================== ==================================

Track Class
===========

========= ===================== =========== ===============================
Attribute Description           Type        Default
========= ===================== =========== ===============================
name      Name of the track     str
program   MIDI program number   int         ``pypianoroll.DEFAULT_PROGRAM``
is_drum   If it is a drum track bool        ``pypianoroll.DEFAULT_IS_DRUM``
pianoroll Downbeat positions    NumPy array ``np.zeros((0, 128))``
========= ===================== =========== ===============================

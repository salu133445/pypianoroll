.. _utilities:

Utilities
=========

Some utility functions are provided. In general, functions listed below will not
modify the input objects and return a copy instead, while methods for
:class:`pypianoroll.Track` and :class:`pypianoroll.Multitrack` may modify the
input objects.

.. autofunction:: pypianoroll.check_pianoroll
.. autofunction:: pypianoroll.assign_constant
.. autofunction:: pypianoroll.binarize
.. autofunction:: pypianoroll.clip
.. autofunction:: pypianoroll.copy
.. autofunction:: pypianoroll.pad
.. autofunction:: pypianoroll.pad_to_multiple
.. autofunction:: pypianoroll.pad_to_same
.. autofunction:: pypianoroll.transpose
.. autofunction:: pypianoroll.trim_trailing_silence

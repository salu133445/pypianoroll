"""
PyPianoroll
=========

Features:
- handle multi-track piano-rolls with metadata
- utilities to manipulate multi-track piano-rolls
- save/load with efficient sparse matrix representation
- write to MIDI file

"""
from .track import Track
from .multitrack import Multitrack
from .pypianoroll import (is_pianoroll, is_standard_pianoroll, binarize, clip,
                          compress_to_active, copy, expand, plot, transpose,
                          trim_trailing_silence)
from .plot import (plot_pianoroll)

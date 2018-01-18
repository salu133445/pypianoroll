"""
Pianoroll
=========

Features:
- handle multi-track piano-rolls with metadata
- utilities to manipulate multi-track piano-rolls
- save/load with efficient sparse matrix representation
- write to MIDI file

"""
from .track import Track
from .multitrack import MultiTrack
from .pianoroll import (binarize, compress_pitch_range, copy, transpose,
                        trim_trailing_silence)

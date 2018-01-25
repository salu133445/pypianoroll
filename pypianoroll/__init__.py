"""
Pypianoroll
===========
A python package for handling multi-track piano-rolls.


Features
========
- handle multi-track piano-rolls with metadata
- utilities to manipulate multi-track piano-rolls
- save/load with efficient sparse matrix representation
- write to MIDI file
"""
from .version import __version__
from .track import Track
from .multitrack import Multitrack
from .pypianoroll import (is_pianoroll, binarize, clip, copy, pad, pad_to_same,
                          plot, transpose, trim_trailing_silence)
from .plot import (plot_pianoroll, save_animation)

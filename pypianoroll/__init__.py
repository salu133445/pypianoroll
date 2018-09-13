"""
Pypianoroll
===========
A python package for handling multi-track pianorolls.

Features
--------
- handle pianorolls of multiple tracks with metadata
- utilities for manipulating pianorolls
- save to and load from .npz files using efficient sparse matrix format
- parse from and write to MIDI files

"""
from __future__ import absolute_import, division, print_function
from pypianoroll.version import __version__
from pypianoroll.track import Track
from pypianoroll.multitrack import Multitrack
from pypianoroll.plot import plot_pianoroll, save_animation
from pypianoroll.utilities import (
    check_pianoroll, assign_constant, binarize, clip, copy, load, pad,
    pad_to_multiple, pad_to_same, parse, plot, save, transpose,
    trim_trailing_silence, write)
import pypianoroll.metrics

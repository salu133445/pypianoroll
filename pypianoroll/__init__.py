"""
Pypianoroll
===========
A python package for handling multitrack pianorolls.

Features
--------
- Manipulate multitrack pianorolls intuitively
- Visualize multitrack pianorolls in a modern DAW like style
- Save and load multitrack pianorolls in space efficient formats
- Parse MIDI files into multitrack pianorolls
- Write multitrack pianorolls into MIDI files

"""
from __future__ import absolute_import, division, print_function

import pypianoroll.metrics
from pypianoroll.multitrack import Multitrack
from pypianoroll.track import Track
from pypianoroll.utilities import (
    assign_constant,
    binarize,
    check_pianoroll,
    clip,
    copy,
    load,
    pad,
    pad_to_multiple,
    pad_to_same,
    parse,
    plot,
    save,
    transpose,
    trim_trailing_silence,
    write,
)
from pypianoroll.version import __version__
from pypianoroll.visualization import (
    plot_multitrack,
    plot_pianoroll,
    plot_track,
    save_animation,
)

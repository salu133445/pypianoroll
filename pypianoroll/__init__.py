"""A Python library for handling multitrack pianorolls.

Pypianoroll is an open source Python library for working with piano
rolls. It provides essential tools for handling multitrack piano rolls,
including efficient I/O as well as manipulation, visualization and
evaluation tools.

Features
--------

- Manipulate multitrack piano rolls intuitively
- Visualize multitrack piano rolls beautifully
- Save and load multitrack piano rolls in a space-efficient format
- Parse MIDI files into multitrack piano rolls
- Write multitrack piano rolls into MIDI files

"""
from . import core, inputs, metrics, multitrack, outputs, track, visualization
from .core import *
from .inputs import *
from .metrics import *
from .multitrack import *
from .outputs import *
from .track import *
from .version import __version__
from .visualization import *

__all__ = ["__version__"]
__all__.extend(core.__all__)
__all__.extend(inputs.__all__)
__all__.extend(metrics.__all__)
__all__.extend(multitrack.__all__)
__all__.extend(outputs.__all__)
__all__.extend(track.__all__)
__all__.extend(visualization.__all__)

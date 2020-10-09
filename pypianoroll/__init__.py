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
from . import core, inputs, metrics, multitrack, track, visualization
from .core import *  # noqa: F401,F403
from .inputs import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403
from .multitrack import *  # noqa: F401,F403
from .track import *  # noqa: F401,F403
from .version import __version__
from .visualization import *  # noqa: F401,F403

__all__ = ["__version__"]
__all__.extend(core.__all__)
__all__.extend(inputs.__all__)
__all__.extend(metrics.__all__)
__all__.extend(multitrack.__all__)
__all__.extend(track.__all__)
__all__.extend(visualization.__all__)

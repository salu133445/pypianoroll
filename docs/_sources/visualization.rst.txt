.. _visualization:

Visualization
=============

Pypianoroll provides utilities for visualizing pianorolls and creating pianoroll
animations.

Note that, by default we assume the pianoroll is using symbolic timing, you may
visualize traditional pianoroll, which use real timing, by setting up x-axis
properties in the plotting functions as well.

Here are some examples.

.. image:: figs/visualization_track.png

.. image:: figs/visualization_multitrack.png

.. image:: figs/visualization_multitrack_closeup.png

Functions
---------

.. autofunction :: pypianoroll.plot
.. autofunction :: pypianoroll.plot_multitrack
.. autofunction :: pypianoroll.plot_track
.. autofunction :: pypianoroll.plot_pianoroll
.. autofunction :: pypianoroll.save_animation

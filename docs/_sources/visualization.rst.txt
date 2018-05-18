.. _visualization:

Visualization
=============

Pypianoroll provides utilities for visualizing piano-rolls and creating
piano-roll animations.

Note that, by default we assume the piano-roll is using symbolic timing, you may
visualize traditional piano-roll, which use real timing, by setting up x-axis
properties in the plotting functions as well.

Here are some examples.

.. image:: figs/visualization_track.png

.. image:: figs/visualization_multitrack.png

.. image:: figs/visualization_multitrack_closeup.png

Related Functions
-----------------

.. autofunction :: pypianoroll.plot
.. autofunction :: pypianoroll.plot_pianoroll
.. autofunction :: pypianoroll.plot.plot_track
.. autofunction :: pypianoroll.plot.plot_multitrack
.. autofunction :: pypianoroll.save_animation

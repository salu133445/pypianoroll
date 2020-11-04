===============
Getting Started
===============

Objects
=======

The main objects in Pypianoroll are :class:`pypianoroll.Multitrack` and :class:`pypianoroll.Track` objects.

A :class:`pypianoroll.Track` object is composed of a piano roll matrix and metadata, including program number, drum indicator and name.

A :class:`pypianoroll.Multitrack` object consists of a list of :class:`pypianoroll.Track` objects, tempo array, downbeat array and metadata.


Examples
========

Here is an example.

.. code-block:: python

    import numpy as np
    from pypianoroll import Multitrack, Track
    from matplotlib import pyplot as plt

    # Create a piano roll matrix, where the first and second axes represent time
    # and pitch, respectively, and assign a C major chord to the piano roll
    pianoroll = np.zeros((96, 128))
    C_maj = [60, 64, 67, 72, 76, 79, 84]
    pianoroll[0:95, C_maj] = 100

    # Create a `pypianoroll.Track` instance
    track = Track(name="my awesome piano", program=0, is_drum=False,
                  pianoroll=pianoroll)

    # Plot the piano roll
    fig, ax = track.plot()
    plt.show()

Here's what the output should look like.

.. image:: images/example_track_plot.png
    :align: center

And here is another example.

.. code-block:: python

    # Extend the pianoroll to demonstrate the usage of down beat array
    track.pianoroll = np.tile(track.pianoroll, (4, 1))
    downbeats = np.zeros(384, bool)
    downbeats[0, 96, 192, 288] = 1

    # Copy the track to demonstrate the usage of `pypianoroll.Multitrack`
    another_track = track.copy()
    another_track.program = 24
    another_track.name = "my awesome guitar"

    # Create a `pypianoroll.Multitrack` instance
    multitrack = Multitrack(name="multitrack", resolution=24, tempo=120.0,
                            downbeat=downbeats, tracks=[track, another_track])

    # Plot the multitrack piano roll
    fig, axs = multitrack.plot()
    plt.show()

Here's what the output should look like.

.. image:: images/example_multitrack_plot.png
    :align: center

Here is how saving and loading works (see `here <save_load.html>`_ for details).

.. code-block:: python

    # Save the `pypianoroll.Multitrack` instance to a .npz file
    multitrack.save('test.npz')

    # Load the .npz file to a `pypianoroll.Multitrack` instance
    loaded = multitrack.load('test.npz')

And here is how to parse and write MIDI files. Pypianoroll currently supports only MIDI files (see `here <parse_write.html>_` for details).

.. code-block:: python

    # Read a MIDI file to a `pypianoroll.Multitrack` instance
    multitrack = multitrack.read('test.mid')

    # Write the `pypianoroll.Multitrack` instance to a MIDI file
    multitrack.write('test.mid')

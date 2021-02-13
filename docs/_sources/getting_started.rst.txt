===============
Getting Started
===============

Welcome to Pypianoroll! We will go through some basic concepts in this tutorial.

.. Hint:: Be sure you have Pypianoroll installed. To install Pypianoroll, please run ``pip install pypianoroll``.

In the following example, we will use `this MIDI file <https://github.com/salu133445/pypianoroll/blob/main/tests/fur-elise.mid>`_ as an example.

First of all, let's import the Pypianoroll library. ::

    import pypianoroll

Now, let's read the example MIDI file into a Multitrack object. ::

    multitrack = pypianoroll.read("fur-elise.mid")
    print(multitrack)

Here's what we got. ::

    Multitrack(name=None, resolution=24, tempo=array(shape=(8976, 1)), downbeat=array(shape=(8976, 1)), tracks=[StandardTrack(name='', program=0, is_drum=False, pianoroll=array(shape=(8976, 128))), StandardTrack(name='', program=0, is_drum=False, pianoroll=array(shape=(8976, 128)))])

You can use dot notation to assess the data. For example, ``multitrack.resolution`` returns the temporal resolution (in time steps per quarter note) and ``multitrack.tracks[0].pianoroll`` returns the piano roll for the first track.

Pypianoroll provides some functions for manipulating the multitrack piano roll. For example, we can trim and binarize the multitrack as follows. ::

    multitrack.trim(0, 12 * multitrack.resolution)
    multitrack.binarize()

Pypianoroll also provides visualization supports. ::

    multitrack.plot()

This will give us the following plot.

.. image:: images/fur-elise.png
    :alt: Piano-roll visualization

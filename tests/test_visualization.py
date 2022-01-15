"""Test cases for visualization functionality."""
from .utils import multitrack


def test_plot_multitrack_separate(multitrack):
    multitrack.plot(mode="separate")


def test_plot_multitrack_blended(multitrack):
    multitrack.plot(mode="blended")


def test_plot_multitrack_hybrid(multitrack):
    multitrack.plot(mode="hybrid")


def test_plot_multitrack_separate_plain(multitrack):
    multitrack.plot(mode="separate", preset="plain")


def test_plot_multitrack_separate_frame(multitrack):
    multitrack.plot(mode="separate", preset="frame")

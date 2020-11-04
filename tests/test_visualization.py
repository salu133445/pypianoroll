"""Test cases for visualization functionality."""
from .utils import multitrack


def test_plot_multitrack_separate(multitrack):
    multitrack.plot(mode="separate")


def test_plot_multitrack_blended(multitrack):
    multitrack.plot(mode="blended")


def test_plot_multitrack_hybrid(multitrack):
    multitrack.plot(mode="hybrid")

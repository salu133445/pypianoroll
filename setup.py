"""Setup script."""
import os.path
from codecs import open

from setuptools import setup

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    LONG_DESCRIPTION = f.read()

VERSION = {}
with open(os.path.join("pypianoroll", "version.py")) as f:
    exec(f.read(), VERSION)

setup(
    name="pypianoroll",
    version=VERSION["__version__"],
    author="Hao-Wen Dong",
    author_email="salu.hwdong@gmail.com",
    description="A toolkit for working with piano rolls",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    download_url=(
        "https://github.com/salu133445/pypianoroll/archive/v"
        + VERSION["__version__"]
        + ".tar.gz"
    ),
    project_urls={
        "Documentation": "https://salu133445.github.io/pypianoroll/"
    },
    keywords=["music", "audio", "music-information-retrieval"],
    packages=["pypianoroll"],
    install_requires=[
        "six>=1.0.0,<2.0",
        "numpy>=1.10.0,<2.0",
        "scipy>=1.0.0,<2.0",
        "pretty_midi>=0.2.8,<1.0",
        "matplotlib>=1.5",
    ],
    extras_require={
        "animation": ["moviepy>=0.2.3.2"],
        "test": ["pytest>=4.6", "pytest-cov>=2.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Multimedia :: Sound/Audio",
    ],
)

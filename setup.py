"""Setup script."""
from pathlib import Path

from setuptools import find_packages, setup


def _get_long_description():
    with open(str(Path(__file__).parent / "README.md"), "r") as f:
        return f.read()


def _get_version():
    with open(str(Path(__file__).parent / "pypianoroll/version.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                delimeter = '"' if '"' in line else "'"
                return line.split(delimeter)[1]
    raise RuntimeError("Cannot read version string.")


VERSION = _get_version()

setup(
    name="pypianoroll",
    version=VERSION,
    author="Hao-Wen Dong",
    author_email="salu.hwdong@gmail.com",
    description="A toolkit for working with piano rolls",
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    download_url=(
        f"https://github.com/salu133445/pypianoroll/archive/v{VERSION}.tar.gz"
    ),
    project_urls={
        "Documentation": "https://salu133445.github.io/pypianoroll/"
    },
    license="MIT",
    keywords=["music", "audio", "music-information-retrieval"],
    packages=find_packages(
        include=["pypianoroll", "pypianoroll.*"], exclude=["tests"]
    ),
    install_requires=[
        "numpy>=1.12.0",
        "scipy>=1.0.0",
        "pretty_midi>=0.2.8",
        "matplotlib>=1.5",
    ],
    extras_require={
        "test": ["pytest>=6.0", "pytest-cov>=2.0"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.6",
)

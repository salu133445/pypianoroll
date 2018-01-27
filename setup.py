from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

exec(compile(open('pypianoroll/version.py', "rb").read(), 'pypianoroll/version.py', 'exec'))

setup(
    name='pypianoroll',
    packages=['pypianoroll'],
    version=__version__,
    description='A python package for handling multi-track piano-rolls.',
    long_description=long_description,
    author='Hao-Wen Dong',
    author_email='salu133445@gmail.com',
    url='https://github.com/salu133445/pypianoroll',
    download_url=('https://github.com/salu133445/pypianoroll/archive/'
                  + __version__ + '.tar.gz'),
    keywords=['music', 'audio', 'piano-roll', 'multi-track'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Multimedia :: Sound/Audio',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=[
        'numpy>=1.10.0',
        'scipy>=1.0.0',
        'matplotlib>=1.5',
        'moviepy>=0.2.3.2',
        'pretty_midi>=0.2.8'
    ],
    project_urls={
        "Documentation": "https://salu133445.github.io/pypianoroll/"
    }
)

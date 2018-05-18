import os.path
from codecs import open
from setuptools import setup

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'README.rst'),
          encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

VERSION = {}
with open(os.path.join('pypianoroll', 'version.py')) as f:
    exec(f.read(), VERSION)

setup(
    name='pypianoroll',
    packages=['pypianoroll'],
    version=VERSION['__version__'],
    description='A python package for handling multi-track piano-rolls.',
    long_description=LONG_DESCRIPTION,
    author='Hao-Wen Dong',
    author_email='salu133445@gmail.com',
    url='https://github.com/salu133445/pypianoroll',
    download_url=('https://github.com/salu133445/pypianoroll/archive/'
                  + VERSION['__version__'] + '.tar.gz'),
    project_urls={"Documentation": "https://salu133445.github.io/pypianoroll/"},
    keywords=['music', 'audio', 'pianoroll', 'multitrack'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Multimedia :: Sound/Audio',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
    setup_requires=[
        'pytest-runner>=2.0,<3.0'
    ],
    install_requires=[
        'six>=1.0.0,<2.0',
        'numpy>=1.10.0,<2.0',
        'scipy>=1.0.0,<2.0',
        'pretty_midi>=0.2.8,<1.0',
    ],
    extras_require={
        'plot':  ['matplotlib>=1.5'],
        'animation': ['moviepy>=0.2.3.2'],
        'pytest':[
            'pytest>=3.0.0,<4.0',
            'pytest-cov>=2.5.1,<3.0',
            'coveralls>=1.1,<2.0',
        ]
    },
    test_suite='tests',
)

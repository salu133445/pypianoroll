import os.path
from codecs import open
from setuptools import setup

readme_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.rst')
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

version = {}
with open(os.path.join('pypianoroll', 'version.py')) as f:
    exec(f.read(), version)

setup(
    name='pypianoroll',
    packages=['pypianoroll'],
    version=version['__version__'],
    description='A python package for handling multi-track piano-rolls.',
    long_description=long_description,
    author='Hao-Wen Dong',
    author_email='salu133445@gmail.com',
    url='https://github.com/salu133445/pypianoroll',
    download_url=('https://github.com/salu133445/pypianoroll/archive/'
                  + version['__version__'] + '.tar.gz'),
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

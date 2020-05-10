"""Synth installer
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='synthingie',  # soon to never be renamed
    version='0.0.1',  # magic needed

    description='An audio synthesys thingie',
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Lucio Tore',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'pyaudio',
        'librosa',
        'ipython',
        'matplotlib',
      ],
)
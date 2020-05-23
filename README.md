# Synthingie

[![Status](http://img.shields.io/travis/luciotorre/synthingie.svg)](https://travis-ci.org/github/luciotorre/synthingie)

A python based audio synth thingie

## Install

We need this package to install the player. Ideas on how to avoid it are welcome.

`$ sudo apt install portaudio19-dev`

Pip install from repo until i do a release.

`$ pip install git+https://github.com/luciotorre/synthingie.git`

If you want widgets to work you need nodejs+npm installed:

`$ sudo apt install npm nodejs`

## Example

```python
from synthingie import Module, Player

SAMPLERATE = 48000
FRAMESIZE = 1024

mod = Module(SAMPLERATE, FRAMESIZE)

osc = mod.sin(2500, 0.2)
gate = mod.naive_square(5, amplitude=0.5) + 0.5

pedestrian = osc * gate

with Player(mod) as p:
    p.play(pedestrian)

```
## Documentation (ja!)

 - [A quick start guide](docs/notebooks/Quickstart.ipynb)
 - [Oscillator gallery](docs/notebooks/Oscillators.ipynb)

### Designing Sound

Exercises from the "Practical Synthetic Sound Design" section of the [book](https://mitpress.mit.edu/books/designing-sound).

 - [24 - Pedestrians](docs/Designing_Sound/24%20-%20Pedestrians.ipynb)


## TODO
 - Stereo
 - More examples
 - Interactivity
 - Visualizers (vumeters, etc)
 - Graphical representation of patches
 - Multi output nodes?
 - Buses
 - Jack compatibility
 - Filters
 - Sequencing
 - Everything else



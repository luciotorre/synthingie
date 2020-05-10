# Synthingie

[![Status](http://img.shields.io/travis/luciotorre/synthingie.svg)](https://travis-ci.org/github/luciotorre/synthingie)

A python based audio synth thingie

# Example

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
# Documentation (ja!)

 - [A quick start guide](docs/notebooks/Quickstart.ipynb)
 - [Oscillator gallery](docs/notebooks/Oscillators.ipynb)

## Designing Sound

Exercises from the "Practical Synthetic Sound Design" section of the [book](https://mitpress.mit.edu/books/designing-sound).

 - [24 - Pedestrians](docs/Designing_Sound/24%20-%20Pedestrians.ipynb)


# TODO
 - Stereo
 - More examples
 - Interactivity
 - Filters
 - Everything else
 
 

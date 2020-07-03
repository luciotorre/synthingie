# Synthingie

[![Status](http://img.shields.io/travis/luciotorre/synthingie.svg)](https://travis-ci.org/github/luciotorre/synthingie)

A python based audio synth thingie

## Install

We need this package to install the player. Ideas on how to avoid it are welcome.

Ubuntu and friends:
`$ sudo apt install portaudio19-dev`

for mac/osx:
`$ brew install portaudio`

Pip install from repo until i do a release.

`$ pip install git+https://github.com/luciotorre/synthingie.git`

If you want the full interactive experience you need to install some bokeh stuff for jupyter lab:

`$ sudo apt install npm nodejs`
`$ pip install jupyterlab`
`$ jupyter labextension install @jupyter-widgets/jupyterlab-manager`
`$ jupyter labextension install @bokeh/jupyter_bokeh`

## Example

```python
import synthingie as st

wave = st.Sin(2500, 0.2)
gate = st.NaiveSquare(5, amplitude=0.5) + 0.5

pedestrian = wave * gate

st.play(pedestrian)

```

## Documentation (ja!)

 - [A quick start guide](docs/notebooks/Quickstart.ipynb)
 - [Oscillator gallery](docs/notebooks/Oscillators.ipynb)

### Designing Sound

Exercises from the "Practical Synthetic Sound Design" section of the [book](https://mitpress.mit.edu/books/designing-sound).

 - [24 - Pedestrians](docs/Designing_Sound/24%20-%20Pedestrians.ipynb)
 - [25 - Phone Tones](docs/Designing_Sound/25%20-%20Phone%20Tones.ipynb)


## TODO

### This Release
 - Notebook examples:
   * Filters
   * live
- move scores to experimental?
- pypi release

### Future
 - Read The Docs docs
 - Videos!
 - Stereo
 - More examples
   * TR-808
 - Instruments with voice management
 - Remote coding
 - Interactivity
   * Fire methods
 - Visualizers:
   * VUmeter
   * spectrogram with history?
 - Graphical representation of patches
 - Filters
   * Butterworth filters
   * Band pass
   * Shelving
 - Effects
 - Sequencing
   * Patterns
 - Integration
   * Jack
   * Midi
   * OSC
   * Pygame
   * Pyglet
- Work more in scores.




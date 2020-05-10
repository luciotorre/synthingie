import os

import numpy as np
import pytest

from synthingie.core import Module, Audio


SAMPLERATE = 48000
FRAMESIZE = 1024


@pytest.fixture
def module():
    return Module(SAMPLERATE, FRAMESIZE)


def test_render(module):
    zero = module.value(0)
    duration = 10.05
    audio = module.render(zero, duration)
    assert audio.samples.shape[0] == int(SAMPLERATE * duration)
    assert np.all(audio.samples == 0)


def test_audio(tmp_path):
    fname = os.path.join(tmp_path, "filename.WAV")
    audio = Audio(SAMPLERATE, np.zeros([SAMPLERATE]))
    audio.save(fname)
    audio2 = Audio.load(fname)
    assert audio.samplerate == audio2.samplerate
    assert all(audio.samples.astype(np.float32) == audio2.samples)

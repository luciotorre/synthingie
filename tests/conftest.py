import pytest

from synthingie import Module


@pytest.fixture
def samplerate():
    return 48000


@pytest.fixture
def framesize(samplerate):
    return int(samplerate / 60)


@pytest.fixture
def module(samplerate, framesize):
    return Module(samplerate, framesize)

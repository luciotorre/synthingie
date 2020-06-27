import pytest


@pytest.fixture
def samplerate():
    return 48000


@pytest.fixture
def framesize(samplerate):
    return int(samplerate / 60)

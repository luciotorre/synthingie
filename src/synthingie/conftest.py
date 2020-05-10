import pytest

from .core import Module, Signal


@pytest.fixture(autouse=True)
def add_basics(doctest_namespace):
    SAMPLERATE = 44800
    FRAMESIZE = 1024

    doctest_namespace["Module"] = Module
    doctest_namespace["Signal"] = Signal
    doctest_namespace["SAMPLERATE"] = SAMPLERATE
    doctest_namespace["FRAMESIZE"] = FRAMESIZE
    doctest_namespace["module"] = Module(SAMPLERATE, FRAMESIZE)

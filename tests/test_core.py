import os
from copy import copy

import numpy as np
import pytest

from synthingie.core import Signal, Audio, signal, Value, DEFAULT_SAMPLERATE


SAMPLERATE = 48000
FRAMESIZE = 1024


def test_pipe():
    def pipein(signal, *args, **kwargs):
        return ("called", signal, args, kwargs)

    s = Signal()
    r = s.pipe(pipein, 1, arg=2)

    assert r == ("called", s, (1,), {'arg': 2})


def test_topological_ordering():
    @signal
    class Left(Signal):
        source: Signal

    @signal
    class Right(Signal):
        source: Signal

    @signal
    class Double(Signal):
        first: Left
        second: Right

    s = Signal()
    s1 = Left(s)
    s2 = Right(s)
    final = Double(s1, s2)

    order = final.topological_sort()
    assert order == [s, s1, s2, final]


def test_new():
    @signal
    class Foo(Signal):
        s: Signal
        i: int

    Value(1)
    Foo(1, 2)
    assert Foo(1, 2) == Foo(Value(1), 2)
    assert Foo(1.1, 2) == Foo(Value(1.1), 2)
    with pytest.raises(ValueError):
        Foo("hello", 2)


def test_render():
    v = Value(1)

    assert all(v.render(1).samples == 1)
    assert len(v.render(2)) == DEFAULT_SAMPLERATE * 2


def test_audio(tmp_path):
    fname = os.path.join(tmp_path, "filename.WAV")
    audio = Audio(SAMPLERATE, np.zeros([SAMPLERATE]))
    audio.save(fname)
    audio2 = Audio.load(fname)
    assert audio.samplerate == audio2.samplerate
    assert all(audio.samples.astype(np.float32) == audio2.samples)


def test_control_method():
    v = Value(2)
    v.set(3)
    v.render_frame()
    assert all(v.output == 3)


def test_copy():
    @signal
    class Foo(Signal):
        a: int
        b: Signal

    v = Foo(1, 1)
    assert v == copy(v)
    assert id(v) != id(copy(v))

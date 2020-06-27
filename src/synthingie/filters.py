from numba.core.decorators import njit
import numpy as np
import numba as nb

import pytest  # NOQA

from .core import Signal, signal
from .oscillators import Sin  # NOQA


@njit([
    nb.types.Tuple((nb.float32, nb.float32))(
        nb.int64, nb.float32, nb.float32, nb.float32[:], nb.float32[:], nb.float32[:]
    )
])
def first_order_low_pass(samplerate, xz1, yz1, signal, frequency, output):
    for n in range(output.shape[0]):
        theta = 2 * np.pi * frequency[n] / samplerate
        gamma = np.cos(theta) / (1 + np.sin(theta))
        a0 = (1 - gamma) / 2
        a1 = a0
        b1 = -gamma

        output[n] = a0 * signal[n] + a1 * xz1 - b1 * yz1

        xz1 = signal[n]
        yz1 = output[n]

    return xz1, yz1


@signal('lop1')
class FirstOrderLowPass(Signal):
    """
    >>> audio = Sin(1000).render(1)
    >>> expected_sin_rms = pytest.approx(1 / np.sqrt(2))
    >>> sin_rms = audio.rms()
    >>> assert sin_rms == expected_sin_rms
    >>> audio = Sin(2048).lop1(1024).render(1)
    >>> assert audio.rms() < sin_rms / 2
    >>> audio = Sin(1024).lop1(2048).render(1)
    >>> assert sin_rms == expected_sin_rms
    """
    signal: Signal
    frequency: Signal

    def setup(self):
        self.xz1 = 0
        self.yz1 = 0

    def __call__(self):
        self.xz1, self.yz1 = first_order_low_pass(
            self.samplerate, self.xz1, self.yz1, self.signal.output,
            self.frequency.output, self.output)


@njit([
    nb.types.Tuple((nb.float32, nb.float32))(
        nb.int64, nb.float32, nb.float32, nb.float32[:], nb.float32[:], nb.float32[:]
    )
])
def first_order_hi_pass(samplerate, xz1, yz1, signal, frequency, output):
    for n in range(output.shape[0]):
        theta = 2 * np.pi * frequency[n] / samplerate
        gamma = np.cos(theta) / (1 + np.sin(theta))
        a0 = (1 + gamma) / 2
        a1 = -a0
        b1 = -gamma

        output[n] = a0 * signal[n] + a1 * xz1 - b1 * yz1

        xz1 = signal[n]
        yz1 = output[n]

    return xz1, yz1


@signal('hip1')
class FirstOrderhiPass(Signal):
    """
    >>> audio = Sin(1000).render(1)
    >>> expected_sin_rms = pytest.approx(1 / np.sqrt(2))
    >>> sin_rms = audio.rms()
    >>> assert sin_rms == expected_sin_rms
    >>> audio = Sin(2048).hip1(1024).render(1)
    >>> assert sin_rms == expected_sin_rms
    >>> audio = Sin(1024).hip1(2048).render(1)
    >>> assert audio.rms() < sin_rms / 2
    """
    signal: Signal
    frequency: Signal

    def setup(self):
        self.xz1 = 0
        self.yz1 = 0

    def __call__(self):
        self.xz1, self.yz1 = first_order_hi_pass(
            self.samplerate, self.xz1, self.yz1, self.signal.output,
            self.frequency.output, self.output)


@njit([
    nb.types.Tuple((nb.float32, nb.float32, nb.float32, nb.float32))(
        nb.int64, nb.float32, nb.float32, nb.float32,
        nb.float32, nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]
    )
])
def second_order_low_pass(samplerate, xz1, yz1, xz2, yz2, signal, cutoff, Q, output):
    for n in range(output.shape[0]):
        theta = 2 * np.pi * cutoff[n] / samplerate
        d = 1 / Q[n]
        beta_p = (d / 2) * np.sin(theta)
        beta = 0.5 * (1 - beta_p) / (1 + beta_p)
        gamma = (0.5 + beta) * np.cos(theta)
        a0 = (0.5 + beta - gamma) / 2.0
        a1 = 0.5 + beta - gamma
        a2 = a0
        b1 = -2 * gamma
        b2 = 2 * beta

        output[n] = a0 * signal[n] + a1 * xz1 - b1 * yz1 + a2 * xz2 - b2 * yz2

        xz2 = xz1
        xz1 = signal[n]
        yz2 = yz1
        yz1 = output[n]

    return xz1, yz1, xz2, yz2


@signal('lop')
class SecondOrderLowPass(Signal):
    """
    >>> audio = Sin(1000).render(1)
    >>> expected_sin_rms = pytest.approx(1 / np.sqrt(2))
    >>> sin_rms = audio.rms()
    >>> assert sin_rms == expected_sin_rms
    >>> audio = Sin(2048).lop(1024).render(1)
    >>> assert audio.rms() < sin_rms / 3
    >>> audio = Sin(1024).lop(2048).render(1)
    >>> assert sin_rms == expected_sin_rms
    """
    signal: Signal
    cutoff: Signal
    Q: Signal = 0.707

    def setup(self):
        self.xz1 = 0
        self.yz1 = 0
        self.xz2 = 0
        self.yz2 = 0

    def __call__(self):
        self.xz1, self.yz1, self.xz2, self.yz2 = second_order_low_pass(
            self.samplerate, self.xz1, self.yz1, self.xz2, self.yz2, self.signal.output,
            self.cutoff.output, self.Q.output, self.output)


@njit([
    nb.types.Tuple((nb.float32, nb.float32, nb.float32, nb.float32))(
        nb.int64, nb.float32, nb.float32, nb.float32,
        nb.float32, nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]
    )
])
def second_order_hi_pass(samplerate, xz1, yz1, xz2, yz2, signal, cutoff, Q, output):
    for n in range(output.shape[0]):
        theta = 2 * np.pi * cutoff[n] / samplerate
        d = 1 / Q[n]
        beta_p = (d / 2) * np.sin(theta)
        beta = 0.5 * (1 - beta_p) / (1 + beta_p)
        gamma = (0.5 + beta) * np.cos(theta)
        a0 = (0.5 + beta + gamma) / 2.0
        a1 = -0.5 - beta - gamma
        a2 = a0
        b1 = -2 * gamma
        b2 = 2 * beta

        output[n] = a0 * signal[n] + a1 * xz1 - b1 * yz1 + a2 * xz2 - b2 * yz2

        xz2 = xz1
        xz1 = signal[n]
        yz2 = yz1
        yz1 = output[n]

    return xz1, yz1, xz2, yz2


@signal('hip')
class SecondOrderHiPass(Signal):
    """
    >>> audio = Sin(1000).render(1)
    >>> expected_sin_rms = pytest.approx(1 / np.sqrt(2))
    >>> sin_rms = audio.rms()
    >>> assert sin_rms == expected_sin_rms
    >>> audio = Sin(2048).hip(1024).render(1)
    >>> assert sin_rms == expected_sin_rms
    >>> audio = Sin(1024).hip(2048).render(1)
    >>> assert audio.rms() < sin_rms / 3
    """
    signal: Signal
    cutoff: Signal
    Q: Signal = 0.707

    def setup(self):
        self.xz1 = 0
        self.yz1 = 0
        self.xz2 = 0
        self.yz2 = 0

    def __call__(self):
        self.xz1, self.yz1, self.xz2, self.yz2 = second_order_hi_pass(
            self.samplerate, self.xz1, self.yz1, self.xz2, self.yz2, self.signal.output,
            self.cutoff.output, self.Q.output, self.output)

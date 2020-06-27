__all__ = ['FixedDelay']

import numpy as np

from .core import Signal, signal
from .misc import Ramp  # NOQA


class CircularBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = np.zeros([self.buffer_size])
        self.buffer_p = 0

    def add(self, samples):
        bz = self.buffer_size
        fz = samples.shape[0]
        self.buffer_p = (self.buffer_p + fz) % bz
        p = self.buffer_p

        idx = min(p, fz)
        self.buffer[p - idx:p] = samples[fz-idx:]
        self.buffer[bz - fz + idx:] = samples[:fz-idx]

    def head(self, out):
        p = self.buffer_p
        fz = out.shape[0]
        bz = self.buffer_size

        idx = min(bz - p, fz)
        out[:idx] = self.buffer[p:min(bz, p + fz)]
        out[idx:] = self.buffer[0:fz - idx]

    def index(self, array):
        return self.buffer[(array + self.buffer_p) % self.buffer_size]


@signal("fixed_delay")
class FixedDelay(Signal):
    """Delay a signal by a fixed amount of time.

    >>> one = Ramp(1)
    >>> delayed = one.fixed_delay(0.5)
    >>> delayed.configure(10, 10)
    >>> more_delayed = one.fixed_delay(1.5)
    >>> more_delayed.configure(10, 10)
    >>> one(); delayed(); more_delayed()
    >>> one.output
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], dtype=float32)
    >>> delayed.output
    array([0. , 0. , 0. , 0. , 0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float32)
    >>> more_delayed.output
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> one(); delayed(); more_delayed()
    >>> more_delayed.output
    array([0. , 0. , 0. , 0. , 0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float32)
    """

    signal: Signal
    delay_s: float

    def setup(self):
        self.buffer_size = int(self.delay_s * self.samplerate) + self.framesize
        self.buffer = CircularBuffer(self.buffer_size)

    def __call__(self):
        self.buffer.add(self.signal.output)
        self.buffer.head(self.output)

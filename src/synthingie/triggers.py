import numpy as np
from numba import njit
from .core import Module, Signal, register
from .table import unicast


@njit(fastmath=True)
def _generate_triggers(framerate, duration, start, data_output):
    end = start

    for i in range(data_output.shape[0]):
        interval = unicast(duration, i) * framerate
        if int(end) == 0:
            data_output[i] = 1
            end = end + interval
        else:
            data_output[i] = 0
        end = end - 1

    return end


@register(Module, "metro")
class MetroSignal(Signal):
    """Generate a trigger every number of seconds.

    >>> mod = Module(10, 10)
    >>> metro = mod.metro(0.5)
    >>> mod.render_frames()
    >>> metro.output
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
    """
    def init(self, rate):
        self.end = 0
        self.rate = rate
        self.output = np.zeros(self.module.framesize, dtype=self.dtype)

    def __call__(self):
        self.end = _generate_triggers(self.samplerate, self.rate, self.end, self.output)


@njit(fastmath=True)
def _generate_cumsum(signal, value, data_output):
    for i in range(data_output.shape[0]):
        value = value + unicast(signal, i)
        data_output[i] = value
    return value


@register(Signal, "cumsum")
class CumsumSignal(Signal):
    """Produce the cumulative sum of the elements.

    >>> mod = Module(10, 10)
    >>> cumsum = mod.metro(0.5).cumsum()
    >>> mod.render_frames()
    >>> cumsum.output
    array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2.])
    >>> mod.render_frames()
    >>> cumsum.output
    array([3., 3., 3., 3., 3., 4., 4., 4., 4., 4.])

    """
    def init(self, signal):
        self.signal = signal.output
        self.value = 0

    def __call__(self):
        self.value = _generate_cumsum(self.signal, self.value, self.output)

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
    >>> mod.render_frame()
    >>> metro.output
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)
    """
    def __init__(self, rate):
        self.end = 0
        self.rate = rate

    def __call__(self):
        self.end = _generate_triggers(self.samplerate, self.rate, self.end, self.output)

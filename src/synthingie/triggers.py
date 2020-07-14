from numba import njit
from .core import Signal, signal
from .table import unicast


@njit()
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


@signal
class Metro(Signal):
    """Generate a trigger every number of seconds.


    >>> metro = Metro(0.5)
    >>> metro.configure(10, 10)
    >>> metro.render_frame()
    >>> metro.output
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)
    """
    rate: float

    def setup(self):
        self.end = 0

    def __call__(self):
        self.end = _generate_triggers(self.samplerate, self.rate, self.end, self.output)

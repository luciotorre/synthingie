from numba.core.decorators import njit
import numpy as np

from .core import Signal, Module, register, SignalTypes, signal_value
from .table import unicast


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
    >>> mod.render_frame()
    >>> cumsum.output
    array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2.], dtype=float32)
    >>> mod.render_frame()
    >>> cumsum.output
    array([3., 3., 3., 3., 3., 4., 4., 4., 4., 4.], dtype=float32)

    """
    def __init__(self, signal: SignalTypes):
        self.signal = signal_value(signal)
        self.value = 0

    def __call__(self):
        self.value = _generate_cumsum(self.signal, self.value, self.output)


@njit
def _make_line(steps, done_frames, line_frames, last_value, samplerate, output):
    if done_frames >= line_frames:
        output.fill(last_value)
        return done_frames, last_value

    for f in range(output.shape[0]):
        if done_frames >= line_frames:
            output[f] = last_value

        delta = 0
        for target, duration_frames in steps:
            if done_frames - delta < duration_frames:
                current_frame = done_frames - delta
                remaining_frames = duration_frames - current_frame
                last_value -= (last_value - target) / remaining_frames
                output[f] = last_value
                break
            else:
                delta += duration_frames

        done_frames += 1

    return done_frames, last_value


@register(Module, "line")
class LineSignal(Signal):
    """Produce the cumulative sum of the elements.

    >>> mod = Module(10, 10)
    >>> line = mod.line()
    >>> line.set([1, 1])
    >>> mod.render_frame()
    >>> line.output
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], dtype=float32)
    >>> line.set([0, 0.5])
    >>> mod.render_frame()
    >>> line.output
    array([0.8, 0.6, 0.4, 0.2, 0. , 0. , 0. , 0. , 0. , 0. ], dtype=float32)
    >>> line.set([1, 0.3], [0, 0.3])
    >>> mod.render_frame()
    >>> line.output
    array([0.33333334, 0.6666667 , 1.        , 0.6666667 , 0.33333334,
           0.        , 0.        , 0.        , 0.        , 0.        ],
          dtype=float32)
    """
    def __init__(self):
        self.steps = None
        self.last_value = 0
        self.done_frames = 0
        self.line_frames = 0

    def set(self, *args):
        """Set the desired values to go to as a series of steps.

        Each argument must be a tuple of (target_value, duration).
        """
        self.steps = np.array(list(
            (value, duration * self.samplerate)
            for (value, duration) in args
        ))
        self.done_frames = 0
        self.line_frames = np.sum(self.steps[:, 1])

    def __call__(self):
        # TODO: make fast
        self.done_frames, self.last_value = \
           _make_line(
               self.steps, self.done_frames,
               self.line_frames, self.last_value,
               self.samplerate, self.output
            )

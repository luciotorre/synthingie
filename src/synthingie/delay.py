import numpy as np

from .core import Signal, register, signal_value


@register(Signal, "fixed_delay")
class FixedDelay(Signal):
    """Delay a signal by a fixed amount of time.

    >>> mod = Module(10, 10)
    >>> one = mod.ramp(0, 1)
    >>> delayed = one.fixed_delay(0.5)
    >>> more_delayed = one.fixed_delay(1.5)
    >>> mod.render_frame()
    >>> one.output
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], dtype=float32)
    >>> delayed.output
    array([0. , 0. , 0. , 0. , 0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float32)
    >>> more_delayed.output
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> mod.render_frame()
    >>> more_delayed.output
    array([0. , 0. , 0. , 0. , 0. , 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float32)
    """
    def __init__(self, signal: Signal, delay_s: float):
        self.signal = signal_value(signal)
        self.delay_s = delay_s
        self.offset = 0

    def bind(self, module):
        super().bind(module)
        self.buffer_size = int(self.delay_s * self.samplerate) + self.framesize
        self.buffer = np.zeros([self.buffer_size])
        self.buffer_p = 0

    def __call__(self):
        self.buffer_p = (self.buffer_p + self.framesize) % self.buffer_size
        p = self.buffer_p
        bz = self.buffer_size
        fz = self.framesize

        idx = min(p, fz)
        self.buffer[p - idx:p] = self.signal[fz-idx:]
        self.buffer[bz - fz + idx:] = self.signal[:fz-idx]

        idx = min(bz - p, fz)
        self.output[:idx] = self.buffer[p:min(bz, p + fz)]
        self.output[idx:] = self.buffer[0:fz - idx]

from numba.core.decorators import njit
import numba as nb
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


@njit([
    nb.float32(
        nb.float32, nb.float32, nb.int64, nb.float32[:]
    )
])
def _make_ramp(last_value, increment_per_second, samplerate, output):
    frame_increment = increment_per_second / samplerate

    for f in range(output.shape[0]):
        last_value += frame_increment
        output[f] = last_value

    return last_value


@register(Module, "ramp")
class RampSignal(Signal):
    """Create ramps, usually to control the amplitude of something else.

    >>> mod = Module(10, 10)
    >>> ramp = mod.ramp(0, 1)
    >>> mod.render_frame()
    >>> ramp.output
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], dtype=float32)
    >>> mod.render_frame()
    >>> ramp.output
    array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. ], dtype=float32)
    >>> ramp.set(1, 2)
    >>> mod.render_frame()
    >>> ramp.output
    array([1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. ], dtype=float32)
    """
    def __init__(self, value, increment_per_second):
        self.set(value, increment_per_second)

    def set(self, value, increment_per_second):
        """Set the ramp at value and increment per second."""
        self.last_value = float(value)
        self.increment_per_second = float(increment_per_second)

    def __call__(self):
        self.last_value = _make_ramp(
            self.last_value, self.increment_per_second, self.samplerate, self.output
        )


@njit([
    nb.types.Tuple((nb.int64, nb.float32))(
        nb.float32[:, :], nb.int64, nb.int64, nb.float32, nb.float32[:]
    )
])
def _make_line(steps, done_frames, line_frames, last_value, output):
    for f in range(output.shape[0]):
        if done_frames > line_frames:
            output[f] = last_value
            continue

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
        else:
            last_value = target
            output[f] = last_value

        done_frames += 1

    return done_frames, last_value


@register(Module, "line")
class LineSignal(Signal):
    """Create ramps, usually to control the amplitude of something else.

    >>> mod = Module(10, 10)
    >>> line = mod.line()
    >>> mod.render_frame()
    >>> line.output
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
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
    >>> line.set([1, 0])
    >>> mod.render_frame()
    >>> line.output
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
    """
    def __init__(self):
        self.steps = np.zeros((0, 0), dtype=self.dtype)
        self.last_value = 0.
        self.done_frames = 0
        self.line_frames = 0

    def set(self, *args):
        """Set the desired values to go to as a series of steps.

        Each argument must be a tuple of (target_value, duration).
        """
        self.steps = np.array(list(
            (value, int(duration * self.samplerate))
            for (value, duration) in args
        ), dtype=self.dtype)
        self.done_frames = 0
        self.line_frames = np.sum(self.steps[:, 1])

    def __call__(self):
        self.done_frames, self.last_value = \
           _make_line(
               self.steps, self.done_frames,
               self.line_frames, self.last_value,
               self.output
            )


ADSR_ATTACK = 0
ADSR_DECAY = 1
ADSR_RELEASE = 2
ADSR_OFF = 3


@njit([
    nb.types.Tuple((nb.float32, nb.float32, nb.int64))(
        nb.int64, nb.float32[:], nb.int64, nb.int64[:], nb.float32, nb.float32[:]
    )
])
def _make_adsr(mode, mode_target, done_frames, mode_frames, last_value, output):
    if mode == ADSR_OFF:
        output.fill(last_value)
        return done_frames, last_value, mode

    remaining_frames = mode_frames[mode] - done_frames
    if remaining_frames == 0:
        increment = 0
    else:
        increment = (mode_target[mode] - last_value) / remaining_frames

    for f in range(output.shape[0]):
        if mode == ADSR_OFF:
            output[f:].fill(last_value)
            break

        if done_frames >= mode_frames[mode]:
            done_frames = 0
            last_value = mode_target[mode]
            # Go to the next mode in sequence, except for decay (no release yet)
            if mode == ADSR_DECAY:
                mode = ADSR_OFF
            else:
                mode += 1

            if mode == ADSR_OFF:
                output[f:].fill(last_value)
                break

            remaining_frames = mode_frames[mode] - done_frames
            if remaining_frames == 0:
                increment = 0
            else:
                increment = (mode_target[mode] - last_value) / remaining_frames

        last_value += increment
        output[f] = last_value
        done_frames += 1

    return done_frames, last_value, mode


@register(Module, "adsr")
class EnvelopeSignal(Signal):
    """Produce an ADSR envelope controlled by the .on() and .off() methods.
    >>> mod = Module(10, 10)
    >>> env = mod.adsr(1, 0.3, 0.3, 0.5, 0.5)
    >>> mod.render_frame()
    >>> env.output
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> env.on()
    >>> mod.render_frame()
    >>> env.output
    array([0.33333334, 0.6666667 , 1.        , 0.8333333 , 0.6666667 ,
           0.5       , 0.5       , 0.5       , 0.5       , 0.5       ],
          dtype=float32)
    >>> mod.render_frame()
    >>> env.output
    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float32)
    >>> env.off()
    >>> mod.render_frame()
    >>> env.output
    array([4.0000001e-01, 3.0000001e-01, 2.0000000e-01, 1.0000000e-01,
           2.7755576e-17, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
           0.0000000e+00, 0.0000000e+00], dtype=float32)


    """
    def __init__(self, level, attack_s, decay_s, sustain, release_s):
        self.level = level
        self.attack_s = attack_s
        self.decay_s = decay_s
        self.sustain = sustain
        self.release_s = release_s

        self.mode = ADSR_OFF
        self.done_frames = 0
        self.last_value = 0

    def bind(self, module):
        super().bind(module)

        sr = self.module.samplerate
        # Parameters in ADSR_ATTACK, ADSR_DECAY, ADSR_RELEASE, ADSR_OFF order
        self.targets = np.array(
            [self.level, self.sustain, 0, 0],
            dtype=np.float32)
        self.frames = np.array([
            self.attack_s * sr, self.decay_s * sr, self.release_s * sr, 0],
            dtype=np.int64)

    def on(self):
        self.mode = ADSR_ATTACK
        self.target = self.level
        self.line_frames = self.attack_s * self.samplerate
        self.done_frames = 0

    def off(self):
        self.mode = ADSR_RELEASE
        self.target = 0.0
        self.line_frames = self.release_s * self.samplerate
        self.done_frames = 0

    def reset(self):
        self.mode = ADSR_OFF
        self.done_frames = 0
        self.last_value = 0.0

    def __call__(self):
        self.done_frames, self.last_value, self.mode = \
            _make_adsr(
               self.mode, self.targets,  self.done_frames,
               self.frames, self.last_value,
               self.output
            )

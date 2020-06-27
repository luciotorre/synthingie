__all__ = ["Ramp", "Line", "ADSR"]

from dataclasses import field

from numba.core.decorators import njit
import numba as nb
import numpy as np

from .core import Signal, signal, control_method
from .table import unicast


@njit(fastmath=True)
def _generate_cumsum(signal, value, data_output):
    for i in range(data_output.shape[0]):
        value = value + unicast(signal, i)
        data_output[i] = value
    return value


@signal("cumsum")
class CumsumSignal(Signal):
    """Produce the cumulative sum of the elements.

    >>> cumsum = Metro(0.5).cumsum()
    >>> cumsum.configure(10, 10)
    >>> cumsum.render_frame()
    >>> cumsum.output
    array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2.], dtype=float32)
    >>> cumsum.render_frame()
    >>> cumsum.output
    array([3., 3., 3., 3., 3., 4., 4., 4., 4., 4.], dtype=float32)

    """
    signal: Signal

    def setup(self):
        self.value = 0

    def __call__(self):
        self.value = _generate_cumsum(self.signal.output, self.value, self.output)


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


@signal
class Ramp(Signal):
    """Create ramps, usually to control the amplitude of something else.

    Args:
        increment_per_second (float): The rate of change per second towards the target.
        start_value (float): The target value of the ramp.

    >>> ramp = Ramp(1)
    >>> ramp.configure(10, 10)
    >>> ramp.render_frame()
    >>> ramp.output
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], dtype=float32)
    >>> ramp.render_frame()
    >>> ramp.output
    array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. ], dtype=float32)
    >>> ramp.set(2, 1)
    >>> ramp.render_frame()
    >>> ramp.output
    array([1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. ], dtype=float32)
    """
    increment_per_second: float
    start_value: float = field(default=0)

    def setup(self):
        self.last_value = self.start_value

    def set(self, increment_per_second: float, start_value: float = 0):
        """Set the ramp at value and increment per second."""
        self.last_value = start_value
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


@signal
class Line(Signal):
    """Create a signal that raises by one every second.

    Usually to control the amplitude of something else.

    >>> line = Line()
    >>> line.configure(10, 10)
    >>> line.render_frame()
    >>> line.output
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> line.set([1, 1])
    >>> line.render_frame()
    >>> line.output
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], dtype=float32)
    >>> line.set([0, 0.5])
    >>> line.render_frame()
    >>> line.output
    array([0.8, 0.6, 0.4, 0.2, 0. , 0. , 0. , 0. , 0. , 0. ], dtype=float32)
    >>> line.set([1, 0.3], [0, 0.3])
    >>> line.render_frame()
    >>> line.output
    array([0.33333334, 0.6666667 , 1.        , 0.6666667 , 0.33333334,
           0.        , 0.        , 0.        , 0.        , 0.        ],
          dtype=float32)
    >>> line.set([1, 0])
    >>> line.render_frame()
    >>> line.output
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
    """
    def setup(self):
        self.steps = np.zeros((0, 0), dtype=self.dtype)
        self.last_value = 0.
        self.done_frames = 0
        self.line_frames = 0

    @control_method
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


@signal
class ADSR(Signal):
    """Produce an ADSR envelope controlled by the .on() and .off() methods.


    >>> env = ADSR(1, 0.3, 0.3, 0.5, 0.5)
    >>> env.configure(10, 10)
    >>> env.render_frame()
    >>> env.output
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> env.on()
    >>> env.render_frame()
    >>> env.output
    array([0.33333334, 0.6666667 , 1.        , 0.8333333 , 0.6666667 ,
           0.5       , 0.5       , 0.5       , 0.5       , 0.5       ],
          dtype=float32)
    >>> env.render_frame()
    >>> env.output
    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float32)
    >>> env.off()
    >>> env.render_frame()
    >>> env.output
    array([4.0000001e-01, 3.0000001e-01, 2.0000000e-01, 1.0000000e-01,
           2.7755576e-17, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
           0.0000000e+00, 0.0000000e+00], dtype=float32)


    """

    level: float
    attack_s: float
    decay_s: float
    sustain: float
    release_s: float

    def setup(self):
        self.mode = ADSR_OFF
        self.done_frames = 0
        self.last_value = 0

        sr = self.samplerate
        # Parameters in ADSR_ATTACK, ADSR_DECAY, ADSR_RELEASE, ADSR_OFF order
        self.targets = np.array(
            [self.level, self.sustain, 0, 0],
            dtype=np.float32)
        self.frames = np.array([
            self.attack_s * sr, self.decay_s * sr, self.release_s * sr, 0],
            dtype=np.int64)

    @control_method
    def on(self):
        self.mode = ADSR_ATTACK
        self.target = self.level
        self.line_frames = self.attack_s * self.samplerate
        self.done_frames = 0

    @control_method
    def off(self):
        self.mode = ADSR_RELEASE
        self.target = 0.0
        self.line_frames = self.release_s * self.samplerate
        self.done_frames = 0

    @control_method
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

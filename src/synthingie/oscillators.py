from dataclasses import field

from numba.core.decorators import njit
from numba import float32, float64, int64
import numpy as np

from .core import Signal, signal
from .oscillator_data import analog_saw, analog_triangle
from .table import Table, unicast


@njit()
def _polyblep(phase, phase_increment):
    """
    inspired by:
      http://research.spa.aalto.fi/publications/papers/smc2010-phaseshaping/phaseshapers.py
      https://www.kvraudio.com/forum/viewtopic.php?t=375517
      http://www.martin-finke.de/blog/articles/audio-plugins-018-polyblep-oscillator/
    """
    phase = phase - np.floor(phase)
    if phase < phase_increment:
        t = phase / phase_increment
        return t+t - t*t - 1.0

    elif phase > 1.0 - phase_increment:
        t = (phase - 1.0) / phase_increment
        return t*t + t+t + 1.0

    else:
        return 0.0


@njit([
    float64(int64, float64, float64, float64, float32[:]),
    float64(int64, float32[:], float64, float64, float32[:]),
    float64(int64, float64, float32[:], float64, float32[:]),
    float64(int64, float32[:], float32[:], float64, float32[:]),
    ], )
def _generate_polyblep_saw(framerate, freq, amplitude, start_phase, data_output):
    end_phase = start_phase

    for i in range(data_output.shape[0]):
        phase_increment = unicast(freq, i) / framerate
        end_phase = start_phase + i * phase_increment
        end_phase = end_phase - np.floor(end_phase)
        p = _polyblep(end_phase, phase_increment)
        data_output[i] = (2 * end_phase - 1 - p) * unicast(amplitude, i)

    end_phase = start_phase + data_output.shape[0] * unicast(freq, i) / framerate
    return end_phase


@njit([
    float64(int64, float64, float64, float64, float64, float32[:]),
    float64(int64, float32[:], float64, float64, float64, float32[:]),
    float64(int64, float64, float32[:], float64, float64, float32[:]),
    float64(int64, float32[:], float32[:], float64, float64, float32[:]),
    float64(int64, float64, float64, float32[:], float64, float32[:]),
    float64(int64, float32[:], float64, float32[:], float64, float32[:]),
    float64(int64, float64, float32[:], float32[:], float64, float32[:]),
    float64(int64, float32[:], float32[:], float32[:], float64, float32[:]),

    ], fastmath=False)
def _generate_polyblep_square(framerate, freq, duty, amplitude, start_phase, data_output):
    end_phase = start_phase

    for i in range(data_output.shape[0]):
        phase_increment = unicast(freq, i) / framerate
        end_phase = start_phase + i * phase_increment

        if end_phase - int(end_phase) < unicast(duty, i):
            naive = unicast(amplitude, i)
        else:
            naive = -unicast(amplitude, i)
        data_output[i] = naive \
            + _polyblep(end_phase, phase_increment) \
            - _polyblep((end_phase + unicast(duty, i)) % 1, phase_increment)

    end_phase = start_phase + data_output.shape[0] * unicast(freq, i) / framerate
    return end_phase


@njit([
    float64(int64, float64, float64, float64, float64, float32[:]),
    float64(int64, float32[:], float64, float64, float64, float32[:]),
    float64(int64, float64, float32[:], float64, float64, float32[:]),
    float64(int64, float32[:], float32[:], float64, float64, float32[:]),
    float64(int64, float64, float64, float32[:], float64, float32[:]),
    float64(int64, float32[:], float64, float32[:], float64, float32[:]),
    float64(int64, float64, float32[:], float32[:], float64, float32[:]),
    float64(int64, float32[:], float32[:], float32[:], float64, float32[:]),

    ], )
def _generate_naive_square(framerate, freq, duty, amplitude, start_phase, data_output):
    end_phase = start_phase

    for i in range(data_output.shape[0]):
        end_phase = start_phase + i * unicast(freq, i) / framerate

        if end_phase - int(end_phase) > unicast(duty, i):
            data_output[i] = unicast(amplitude, i)
        else:
            data_output[i] = -unicast(amplitude, i)

    end_phase = start_phase + data_output.shape[0] * unicast(freq, i) / framerate
    return end_phase


@signal
class Saw(Signal):
    """Generate a saw wave at the specified frequency and amplitude.

    >>> wave = Saw(440, 2)
    >>> wave.render_frame()
    """

    frequency: Signal
    amplitude: Signal = field(default=1)

    def setup(self):
        # we need to track phase, so consecutive samples end and start in the same places + 1
        self.phase = 0.0

    def __call__(self):
        self.phase = _generate_polyblep_saw(
            self.samplerate, self.frequency.output, self.amplitude.output, self.phase, self.output)


@signal
class NaiveSquare(Signal):
    """Generate a naive square wave at the specified frequency and amplitude.

    >>> wave = NaiveSquare(440, 2)
    >>> wave.render_frame()
    """
    frequency: Signal
    amplitude: Signal = field(default=1)
    duty: Signal = field(default=0.5)
    phase: float = field(default=0)

    def __call__(self):
        self.phase = _generate_naive_square(
            self.samplerate, self.frequency.output, self.duty.output, self.amplitude.output, self.phase, self.output)


@signal
class Square(NaiveSquare):
    """Generate a square wave at the specified frequency and amplitude.

    >>> wave = Square(440, 2)
    >>> wave.render_frame()
    """

    def __call__(self):
        self.phase = _generate_polyblep_square(
            self.samplerate, self.frequency.output, self.duty.output, self.amplitude.output, self.phase, self.output)


@signal
class TableSignal(Signal):
    TABLE_SIZE = 8192

    frequency: Signal
    amplitude: Signal = field(default=1)
    phase: float = field(default=0)

    def setup(self):
        self.generator = self.build_table()

    def __call__(self):
        self.phase = self.generator.generate(self.frequency.output, self.amplitude.output, self.phase, self.output)


@signal
class Sin(TableSignal):
    """Generate a sin wave at the specified frequency and amplitude.

    >>> sin = Sin(440, 2)
    >>> sin.render_frame()
    """
    def build_table(self):
        return Table(
            np.sin(np.linspace(0, 2 * np.pi, self.TABLE_SIZE, endpoint=False), dtype=self.dtype),
            self.samplerate
        )


@signal
class NaiveSaw(TableSignal):
    """Generate a naive saw wave at the specified frequency and amplitude.

    >>> naive_saw = NaiveSaw(440, 2)
    >>> naive_saw.render_frame()
    """

    def build_table(self):
        return Table(
            np.linspace(0, 1, self.TABLE_SIZE, dtype=self.dtype),
            self.samplerate)


@signal
class NaiveTriangle(TableSignal):
    """Generate a naive triangle wave at the specified frequency and amplitude.

    >>> wave = NaiveTriangle(440, 2)
    >>> wave.render_frame()
    """
    def build_table(self):
        return Table(
            np.concatenate([
                np.linspace(0, 1, self.TABLE_SIZE // 4, endpoint=False, dtype=self.dtype),
                np.linspace(1, -1, self.TABLE_SIZE // 2, endpoint=False, dtype=self.dtype),
                np.linspace(-1, 0, self.TABLE_SIZE // 4, endpoint=False, dtype=self.dtype),
            ]), self.samplerate)


Triangle = NaiveTriangle


@signal
class AnalogSin(TableSignal):
    """Generate a analog-like sin wave at the specified frequency and amplitude.

    Analog sin is the "quadratic aproximation of sine with slightly richer harmonics"
    https://github.com/VCVRack/Fundamental/blob/v0.6/src/VCO.cpp

    >>> sin = AnalogSin(440, 2)
    >>> sin.render_frame()
    """
    def build_table(self):
        def analog_sin(phase):
            if phase < 0.5:
                v = 1 - 16 * ((phase - 0.25) ** 2)
            else:
                v = -1 + 16 * ((phase - 0.75) ** 2)
            return v * 1.08

        return Table(
            np.array([analog_sin(x) for x in np.linspace(0, 1, 2048, endpoint=False)]),
            self.samplerate
        )


@signal
class AnalogSaw(TableSignal):
    """Generate a analog-like saw wave at the specified frequency and amplitude.

    >>> wave = AnalogSaw(440, 2)
    >>> wave.render_frame()
    """
    def build_table(self):
        return Table(analog_saw, self.samplerate)


@signal
class AnalogTriangle(TableSignal):
    """Generate a analog-like triangle wave at the specified frequency and amplitude.

    >>> wave = AnalogTriangle(440, 2)
    >>> wave.render_frame()
    """
    def build_table(self):
        return Table(analog_triangle, self.samplerate)


@signal
class WhiteNoise(Signal):
    """
    >>> noise = WhiteNoise()
    >>> noise.render_frame()
    """
    def __call__(self):
        self.output[:] = np.random.rand(self.output.shape[0]) - 0.5

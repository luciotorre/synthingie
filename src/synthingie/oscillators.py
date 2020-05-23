from numba.core.decorators import njit
from numba import float32, float64, int64
import numpy as np

from .core import Signal, Module, register, SignalTypes, signal_value
from .oscillator_data import analog_saw, analog_triangle
from .table import Table, unicast


@njit(fastmath=True)
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
    ], fastmath=True)
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

    ], fastmath=True)
def _generate_polyblep_square(framerate, freq, duty, amplitude, start_phase, data_output):
    end_phase = start_phase

    for i in range(data_output.shape[0]):
        phase_increment = unicast(freq, i) / framerate
        end_phase = start_phase + i * phase_increment

        if end_phase - int(end_phase) >= unicast(duty, i):
            data_output[i] = unicast(amplitude, i) - (
                unicast(amplitude, i) * _polyblep(end_phase, phase_increment)
            )
        else:
            data_output[i] = -unicast(amplitude, i) + (
                unicast(amplitude, i) * _polyblep(end_phase + unicast(duty, i), phase_increment)
            )

    end_phase = start_phase + data_output.shape[0] * unicast(freq, i) / framerate
    return end_phase


@njit(fastmath=True)
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


@register(Module, "saw")
class PolyBlepSaw(Signal):
    """Generate a saw wave at the specified frequency and amplitude.

    >>> wave = module.saw(440, 2)
    >>> module.render_frame()
    """
    def __init__(self, frequency: SignalTypes, amplitude: SignalTypes = 1.0):
        # Store the arguments
        self.frequency = signal_value(frequency)
        self.amplitude = signal_value(amplitude)
        # we need to track phase, so consecutive samples end and start in the same places + 1
        self.phase = 0.0

    def __call__(self):
        self.phase = _generate_polyblep_saw(
            self.samplerate, self.frequency, self.amplitude, self.phase, self.output)


@register(Module, "naive_square")
class NaiveSquare(Signal):
    """Generate a naive square wave at the specified frequency and amplitude.

    >>> wave = module.naive_square(440, 2)
    >>> module.render_frame()
    """
    def __init__(self, frequency: SignalTypes, duty: SignalTypes = 0.5, amplitude: SignalTypes = 1.0):
        # Store the arguments
        self.frequency = signal_value(frequency)
        self.amplitude = signal_value(amplitude)
        self.duty = signal_value(duty)
        # we need to track phase, so consecutive samples end and start in the same places + 1
        self.phase = 0.0

    def __call__(self):
        self.phase = _generate_naive_square(
            self.samplerate, self.frequency, self.duty, self.amplitude, self.phase, self.output)


@register(Module, "square")
class PolyBlepSquare(NaiveSquare):
    """Generate a square wave at the specified frequency and amplitude.

    >>> wave = module.square(440, 2)
    >>> module.render_frame()
    """

    def __call__(self):
        self.phase = _generate_polyblep_square(
            self.samplerate, self.frequency, self.duty, self.amplitude, self.phase, self.output)


class TableSignal(Signal):
    TABLE_SIZE = 8192

    def __init__(self, frequency: SignalTypes, amplitude: SignalTypes = 1.0):
        # Store the arguments
        self.frequency = signal_value(frequency)
        self.amplitude = signal_value(amplitude)
        # we need to track phase, so consecutive samples end and start in the same places + 1
        self.phase = 0.0

    def bind(self, module):
        super().bind(module)
        # create table, fill it with something
        self.generator = self.build_table()

    def __call__(self):
        self.phase = self.generator.generate(self.frequency, self.amplitude, self.phase, self.output)


@register(Module, "sin")
class Sin(TableSignal):
    """Generate a sin wave at the specified frequency and amplitude.

    >>> sin = module.sin(440, 2)
    >>> module.render_frame()
    """
    def build_table(self):
        return Table(
            np.sin(np.linspace(0, 2 * np.pi, self.TABLE_SIZE, endpoint=False), dtype=self.dtype),
            self.samplerate
        )


@register(Module, "naive_saw")
class NaiveSaw(TableSignal):
    """Generate a naive saw wave at the specified frequency and amplitude.

    >>> naive_saw = module.naive_saw(440, 2)
    >>> module.render_frame()
    """

    def build_table(self):
        return Table(
            np.linspace(0, 1, self.TABLE_SIZE, dtype=self.dtype),
            self.samplerate)


@register(Module, "naive_triangle")
class NaiveTriangle(TableSignal):
    """Generate a naive triangle wave at the specified frequency and amplitude.

    >>> wave = module.naive_triangle(440, 2)
    >>> module.render_frame()
    """
    def build_table(self):
        return Table(
            np.concatenate([
                np.linspace(0, 1, self.TABLE_SIZE // 4, endpoint=False, dtype=self.dtype),
                np.linspace(1, -1, self.TABLE_SIZE // 2, endpoint=False, dtype=self.dtype),
                np.linspace(-1, 0, self.TABLE_SIZE // 4, endpoint=False, dtype=self.dtype),
            ]), self.samplerate)


@register(Module, "analog_sin")
class AnalogSin(TableSignal):
    """Generate a analog-like sin wave at the specified frequency and amplitude.

    Analog sin is the "quadratic aproximation of sine with slightly richer harmonics"
    https://github.com/VCVRack/Fundamental/blob/v0.6/src/VCO.cpp

    >>> sin = module.analog_sin(440, 2)
    >>> module.render_frame()
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


@register(Module, "analog_saw")
class AnalogSaw(TableSignal):
    """Generate a analog-like saw wave at the specified frequency and amplitude.

    >>> wave = module.analog_saw(440, 2)
    >>> module.render_frame()
    """
    def build_table(self):
        return Table(analog_saw, self.samplerate)


class AnalogTriangle(TableSignal):
    """Generate a analog-like triangle wave at the specified frequency and amplitude.

    >>> wave = module.triangle(440, 2)
    >>> module.render_frame()
    """
    def build_table(self):
        return Table(analog_triangle, self.samplerate)


register(Module, "analog_triangle")(AnalogTriangle)
register(Module, "triangle")(AnalogTriangle)

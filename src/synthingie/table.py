from numba.core.decorators import njit, generated_jit
from numba import types, float64, float32, int64
import numpy as np


def is_power_of_two(value):
    return value and (value & (value - 1)) == 0


@generated_jit(nopython=True)
def unicast(source, index):
    if isinstance(source, types.Float) or isinstance(source, types.Integer):
        return lambda source, index: source
    else:
        return lambda source, index: source[index]


@njit([
    float64(float32[:], int64, float64, float64, float64, float32[:]),
    float64(float32[:], int64, float32[:], float64, float64, float32[:]),
    float64(float32[:], int64, float64, float32[:], float64, float32[:]),
    float64(float32[:], int64, float32[:], float32[:], float64, float32[:]),
], )
def _generate_table(data, framerate, freq, amplitude, start_phase, data_output):
    samples = data.shape[0]
    end_phase = start_phase
    mask = samples - 1

    for i in range(data_output.shape[0]):
        # linear interpolation
        # get first and next value
        sample_pointer = samples * end_phase
        spi = int(sample_pointer) & mask
        v1 = data[spi]
        v2 = data[(spi + 1) & mask]

        # how much in the middle
        frac = sample_pointer - np.floor(sample_pointer)
        data_output[i] = (v1 + (v2 - v1) * frac) * unicast(amplitude, i)
        end_phase = end_phase + unicast(freq, i) / framerate

    return end_phase


class Table:
    def __init__(self, data, samplerate):
        assert is_power_of_two(data.shape[0])
        assert len(data.shape) == 1

        self.data = data.astype(np.float32)
        self.samplerate = samplerate

    def generate(self, freq, amplitude, start_phase, output):
        return _generate_table(self.data, self.samplerate, freq, amplitude, start_phase, output)

import numpy as np
from scipy.signal import find_peaks

from pytest import approx

from synthingie import table

sr = 8192


def test_table():
    source = np.linspace(0, 1, sr)
    t = table.Table(source, sr)
    output = np.ones([sr])

    t.generate(1, 1, 0, output)
    assert max(output) == 1.0
    assert sum(output - source) == 0

    # double amplitude
    t.generate(1, 2, 0, output)
    assert max(output) == 2.0

    # double amplitude, but half frequency
    t.generate(0.5, 2, 0, output)
    assert max(output) == approx(1.0, 0.001)

    # 8.5 freq -> 8 peaks
    t.generate(8.5, 1, 0, output)
    assert len(find_peaks(output)[0]) == 8


def test_table_interpolation():
    source = np.linspace(0, 1, 2 * sr)
    t = table.Table(source, sr)
    output = np.ones([sr])

    t.generate(1, 1, 0, output)
    assert max(output) == approx(1.0, 0.0001)
    assert max(output - source[::2]) == 0

    assert sum(output) == approx(output.shape[0] / 2, 1)

import numpy as np

from synthingie import table


def test_table_benchmark(benchmark, samplerate, framesize):
    source = np.linspace(0, 1, 4096)
    t = table.Table(source, samplerate)
    output = np.ones([framesize], dtype=np.float32)

    benchmark(t.generate, 1., 1., 0., output)

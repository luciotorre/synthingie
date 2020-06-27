import synthingie.oscillators as osc


def test_fixed_delay(benchmark):
    sin = osc.Sin(440)
    delayed = sin.fixed_delay(10)

    delayed.configure()
    benchmark(delayed)

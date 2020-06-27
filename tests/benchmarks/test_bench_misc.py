from synthingie import misc


def test_line(benchmark):
    line = misc.Line()
    line.configure()
    line.set((1, 1))
    benchmark(line)


def test_adsr(benchmark):
    adsr = misc.ADSR(1, 0.5, 0.5, 0.5, 1)
    adsr.configure()
    adsr.on()
    benchmark(adsr)


def test_ramp(benchmark):
    ramp = misc.Ramp(1, 1)
    ramp.configure()
    benchmark(ramp)

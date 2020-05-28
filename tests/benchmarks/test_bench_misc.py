
def test_line(module, benchmark):
    line = module.line()
    line.set((1, 1))
    benchmark(line)


def test_adsr(module, benchmark):
    adsr = module.adsr(1, 0.5, 0.5, 0.5, 1)
    adsr.on()
    benchmark(adsr)


def test_ramp(module, benchmark):
    ramp = module.ramp(1, 1)
    benchmark(ramp)

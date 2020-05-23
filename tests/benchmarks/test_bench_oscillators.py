
def test_sin(module, benchmark):
    sin = module.sin(440)
    benchmark(sin)


def test_saw(module, benchmark):
    sin = module.saw(440)
    benchmark(sin)


def test_square(module, benchmark):
    sin = module.square(440)
    benchmark(sin)


def test_naive_square(module, benchmark):
    sin = module.naive_square(440)
    benchmark(sin)

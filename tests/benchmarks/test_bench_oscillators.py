import synthingie.oscillators as osc


def test_sin(benchmark):
    sin = osc.Sin(440)
    sin.configure()
    benchmark(sin)


def test_saw(benchmark):
    saw = osc.Saw(440)
    saw.configure()
    benchmark(saw)


def test_square(benchmark):
    sq = osc.Square(440)
    sq.configure()
    benchmark(sq)


def test_naive_square(benchmark):
    sq = osc.NaiveSquare(440)
    sq.configure()
    benchmark(sq)


def test_white_noise(benchmark):
    noise = osc.WhiteNoise()
    noise.configure()
    benchmark(noise)

import synthingie.oscillators as osc


def test_lop1(benchmark):
    sin = osc.Sin(440)
    low = sin.lop1(440)

    low.configure()
    benchmark(low)


def test_hip1(benchmark):
    sin = osc.Sin(440)
    hi = sin.hip1(440)

    hi.configure()
    benchmark(hi)


def test_lop(benchmark):
    sin = osc.Sin(440)

    low = sin.lop(440)

    low.render_frame()
    benchmark(low)


def test_hip(benchmark):
    sin = osc.Sin(440)

    hi = sin.hip(440)

    hi.render_frame()
    benchmark(hi)

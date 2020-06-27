from synthingie.oscillators import Sin


def test_plus(benchmark):
    one = Sin(440)
    plus = one + one
    plus.configure()
    benchmark(plus)


def test_times(benchmark):
    one = Sin(440)
    times = one * one
    times.configure()
    benchmark(times)


def test_mod(benchmark):
    one = Sin(440)
    mod = one % one
    mod.configure()
    benchmark(mod)


def test_less(benchmark):
    one = Sin(440)
    less = one - one
    less.configure()
    benchmark(less)


def test_gt(benchmark):
    one = Sin(440)
    gt = one > one
    gt.configure()
    benchmark(gt)


def test_abs(benchmark):
    one = Sin(440)
    abs_one = abs(one)
    abs_one.configure()
    benchmark(abs_one)


def test_pow(benchmark):
    one = Sin(440)
    pw = one ** one
    pw.configure()
    benchmark(pw)

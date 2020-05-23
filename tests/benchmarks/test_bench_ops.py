
def test_plus(module, benchmark):
    one = module.sin(440)
    plus = one + one
    benchmark(plus)


def test_times(module, benchmark):
    one = module.sin(440)
    times = one * one
    benchmark(times)


def test_mod(module, benchmark):
    one = module.sin(440)
    mod = one % one
    benchmark(mod)


def test_less(module, benchmark):
    one = module.sin(440)
    less = one - one
    benchmark(less)


def test_gt(module, benchmark):
    one = module.sin(440)
    gt = one > one
    benchmark(gt)

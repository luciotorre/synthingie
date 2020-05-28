
def test_fixed_delay(module, benchmark):
    sin = module.sin(440)
    delayed = sin.fixed_delay(10)
    benchmark(delayed)


def test_line(module, benchmark):
    line = module.line()
    line.set((1, 1))
    benchmark(line)


def test_line(module, benchmark):
    line = module.line()
    benchmark.pedantic(line, setup=lambda: line.set((1, 1)))

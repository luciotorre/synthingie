
# mostly make sure oscillators work with type combinations


def test_saw(module):
    saw = module.saw(440)
    module.render_frame()

    saw = module.saw(module.sin(2), module.sin(2))  # NOQA
    module.render_frame()


def test_square(module):
    square = module.square(440)
    module.render_frame()

    square = module.square(module.sin(2), module.sin(2))  # NOQA
    module.render_frame()


def test_naive_square(module):
    naive_square = module.naive_square(440)
    module.render_frame()

    naive_square = module.naive_square(module.sin(2), module.sin(2))  # NOQA
    module.render_frame()

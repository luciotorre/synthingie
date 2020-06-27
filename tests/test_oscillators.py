
# mostly make sure oscillators work with type combinations
import synthingie.oscillators as osc


def test_saw():
    saw = osc.Saw(440)
    saw.render_frame()

    saw = osc.Saw(osc.Sin(2), osc.Sin(2))
    saw.render_frame()


def test_square():
    square = osc.Square(440)
    square.render_frame()

    square = osc.Square(osc.Sin(2), osc.Sin(2))
    square.render_frame()


def test_naive_square():
    naive_square = osc.NaiveSquare(440)
    naive_square.render_frame()

    naive_square = osc.NaiveSquare(osc.Sin(2), osc.Sin(2))
    naive_square.render_frame()

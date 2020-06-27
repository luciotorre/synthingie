import pytest

from synthingie import misc


def test_fixed_delay():
    ramp = misc.Ramp(0, 1)
    delayed = ramp.fixed_delay(1)

    delayed.configure()

    for i in range(int(2 * delayed.samplerate / delayed.framesize)):
        delayed.render_frame()
        assert pytest.approx(delayed.output[0], max(0, ramp.output[0] - 1))

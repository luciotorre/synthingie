import pytest


def test_fixed_delay(module):
    ramp = module.ramp(0, 1)
    delayed = ramp.fixed_delay(1)

    for i in range(int(2 * module.samplerate / module.framesize)):
        module.render_frame()
        assert pytest.approx(delayed.output[0], max(0, ramp.output[0] - 1))

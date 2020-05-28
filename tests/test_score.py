from synthingie import score


def test_wait():
    w = score.Wait(10)

    sc = w.run()
    finished, elapsed = sc.advance(5.)
    assert finished is False
    assert elapsed == 5.0

    finished, elapsed = sc.advance(10.)
    assert finished is True
    assert elapsed == 5.0


def test_call():
    calls = []

    def callback(*args, **kwargs):
        calls.append((args, kwargs))

    f = score.Call(callback, 1, 2, a=3)

    finished, elapsed = f.run().advance(10)
    assert finished is True
    assert elapsed == 0.0
    assert calls == [((1, 2), {'a': 3})]


def test_sequence():
    calls = []

    def callback(label):
        calls.append(label)

    sc = score.Call(callback, "start") + score.Wait(10) + score.Call(callback, "end")
    rsc = sc.run()

    finished, elapsed = rsc.advance(5.)
    assert finished is False
    assert elapsed == 5.0
    assert calls == ['start']

    finished, elapsed = rsc.advance(10.)
    assert finished is True
    assert elapsed == 5.0
    assert calls == ['start', 'end']


def test_sequence_multi():
    calls = []

    def callback(label):
        calls.append(label)

    sc = score.Call(callback, "start") + score.Wait(10)
    sc = score.Sequence([sc, sc, sc])
    rsc = sc.run()

    finished, elapsed = rsc.advance(5.)
    assert finished is False
    assert elapsed == 5.0
    assert calls == ['start']

    finished, elapsed = rsc.advance(10.)
    assert finished is False
    assert elapsed == 10.0

    finished, elapsed = rsc.advance(16.)
    assert finished is True
    assert elapsed == 15.0
    assert calls == ['start'] * 3


def test_repeat():
    calls = []

    def callback(label):
        calls.append(label)

    sc = (
        score.Wait(0.5) +
        score.Call(callback, "step") +
        score.Wait(0.5)
    ) * 10

    rsc = sc.run()

    finished, elapsed = rsc.advance(5.)

    assert finished is False
    assert elapsed == 5.0
    assert calls == ['step'] * 5

    finished, elapsed = rsc.advance(10.)
    assert finished is True
    assert elapsed == 5.0
    assert calls == ['step'] * 10


def test_time_warp():
    calls = []

    def callback(label):
        calls.append(label)

    def build_score():
        return (
            score.Wait(0.5) +
            score.Call(callback, "step") +
            score.Wait(0.5)
        ) * 10

    # normal time flow
    sc = build_score()
    rsc = sc.run()

    finished, elapsed = rsc.advance(5.)
    assert finished is False
    assert elapsed == 5.0
    assert calls == ['step'] * 5

    # double time flow
    calls[:] = []
    sc = build_score()
    sc = score.TimeWarp(
        sc,
        lambda t: 2. * t,
        lambda t: t / 2.,
    )
    rsc = sc.run()

    finished, elapsed = rsc.advance(2.5)
    assert elapsed == 2.5
    finished, elapsed = rsc.advance(2.5)
    assert elapsed == 2.5
    assert finished is False
    assert calls == ['step'] * 5 * 2


def test_overrun():
    calls = []

    def callback(label):
        calls.append(label)

    sc = score.Call(callback, "hi") + score.Wait(1) + score.Call(callback, "bye")
    rsc = sc.run()

    rsc.advance(1)
    rsc.advance(1)
    rsc.advance(1)
    rsc.advance(1)
    rsc.advance(1)
    rsc.advance(1)
    rsc.advance(1)
    rsc.advance(1)

    assert calls == ["hi", "bye"]


def test_advance_cero():
    calls = []

    def callback(label):
        calls.append(label)

    sc = score.Call(callback, "hi") + score.Wait(1) + score.Call(callback, "bye")
    rsc = sc.run()

    rsc.advance(0)
    assert calls == ["hi"]
    rsc.advance(1)
    assert calls == ["hi", "bye"]

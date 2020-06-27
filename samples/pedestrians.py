import synthingie.oscillators as osc
from synthingie.player import play


def pedestrian():
    wave = osc.Sin(2500, 0.2)
    gate = osc.NaiveSquare(5, amplitude=0.5) + 0.5

    pedestrian = wave * gate

    play(pedestrian)


if __name__ == "__main__":
    pedestrian()

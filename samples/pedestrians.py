from synthingie import Module, Player


SAMPLERATE = 48000
FRAMESIZE = 1024

mod = Module(SAMPLERATE, FRAMESIZE)

osc = mod.sin(2500, 0.2)
gate = mod.naive_square(5, amplitude=0.5) + 0.5

pedestrian = osc * gate

with Player(mod) as p:
    p.play(pedestrian)

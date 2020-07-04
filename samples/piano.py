import curses

from mingus.containers import Note

import synthingie as st

piano = """

   ┌───▄▄▄▄▄─▄▄▄▄▄───┬───▄▄▄▄▄─▄▄▄▄▄─▄▄▄▄▄───┐
   │   █████ █████   │   █████ █████ █████   │
   │   █████ █████   │   █████ █████ █████   │
   │   █████ █████   │   █████ █████ █████   │
   │   █████ █████   │   █████ █████ █████   │
   │   █████ █████   │   █████ █████ █████   │
   │   █▀▀▀█ █▀▀▀█   │   █▀▀▀█ █▀▀▀█ █▀▀▀█   │
   │   █ w █ █ e █   │   █ t █ █ y █ █ u █   │
   │   █▄▄▄█ █▄▄▄█   │   █▄▄▄█ █▄▄▄█ █▄▄▄█   │
   │   █████ █████   │   █████ █████ █████   │
   │     │     │     │     │     │     │     │
   │     │     │     │     │     │     │     │
   │     │     │     │     │     │     │     │
   │  a  │  s  │  d  │  f  │  g  │  h  │  j  │
   └─────┴─────┴─────┴─────┴─────┴─────┴─────┘

"""

piano_map = {
    'a': Note("C"),
    'w': Note("C#"),
    's': Note("D"),
    'e': Note("D#"),
    'd': Note("E"),
    'f': Note("F"),
    't': Note("F#"),
    'g': Note("G"),
    'y': Note("G#"),
    'h': Note("A"),
    'u': Note("A#"),
    'j': Note("B"),
}


def main():
    window = curses.initscr()
    window.addstr(piano)

    frequency = st.Value(Note("C").to_hertz())
    wave = st.Sin(frequency)
    envelope = st.ADSR(1, 0.1, 0.1, 0.7, 0.3)
    envelope.on()
    instrument = wave * envelope

    player = st.player.Player()
    player.start(instrument)

    try:
        curses.cbreak()
        curses.noecho()
        curses.curs_set(0)

        window.redrawwin()
        window.refresh()

        while True:
            c = chr(window.getch())

            if c in piano_map:
                note = piano_map[c]
                frequency.set(note.to_hertz())
                envelope.on()
            else:
                envelope.off()

            window.redrawwin()
            window.refresh()

    finally:
        curses.endwin()


if __name__ == "__main__":
    main()

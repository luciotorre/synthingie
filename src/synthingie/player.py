import pyaudio
import time
import numpy as np

from synthingie.core import DEFAULT_SAMPLERATE, DEFAULT_FRAMESIZE


class Player:
    def __init__(self, samplerate=DEFAULT_SAMPLERATE, framesize=DEFAULT_FRAMESIZE):
        self.samplerate = samplerate
        self.framesize = framesize
        self.stream = None
        self.output = np.zeros([self.framesize], dtype=np.float32)

    def init(self):
        self.stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(4),
            channels=1,
            rate=self.samplerate,
            frames_per_buffer=self.framesize,
            output=True, stream_callback=self.callback
        )

    def play(self, target):
        self.pyaudio = pyaudio.PyAudio()
        try:
            self.target = target
            self.steps = target.topological_sort()
            target.configure(self.samplerate, self.framesize)
            self.init()
            self.stream.start_stream()

            while self.stream.is_active():
                time.sleep(1)
            self.stream.stop_stream()
        finally:
            if self.stream:
                self.stream.close()

            self.pyaudio.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        self.target.render_frame(self.steps)
        self.output[:] = self.target.output
        return (self.output, pyaudio.paContinue)


def play(signal):
    p = Player()
    p.play(signal)


if __name__ == "__main__":
    import synthingie
    sin = synthingie.oscillators.Sin(440)
    play(sin)

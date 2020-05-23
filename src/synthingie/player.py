import pyaudio
import time
import numpy as np


class Player:
    def __init__(self, module):
        self.module = module
        self.stream = None
        self.output = np.zeros([self.module.framesize], dtype=np.float32)

    def __enter__(self):
        self.pyaudio = pyaudio.PyAudio()
        return self

    def __exit__(self, *args):
        if self.stream:
            self.stream.close()

        self.pyaudio.terminate()

    def init(self):
        self.stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(4),
            channels=1,
            rate=self.module.samplerate,
            frames_per_buffer=self.module.framesize,
            output=True, stream_callback=self.callback
        )

    def play(self, target):
        self.target = target
        self.init()
        self.stream.start_stream()

        while self.stream.is_active():
            time.sleep(1)
        self.stream.stop_stream()

    def callback(self, in_data, frame_count, time_info, status):
        self.module.render_frame()
        self.output[:] = self.target.output
        return (self.output, pyaudio.paContinue)


if __name__ == "__main__":
    import synthingie
    mod = synthingie.Module(48000, 1024)
    sin = mod.sin(440)
    with Player(mod) as p:
        p.play(sin)

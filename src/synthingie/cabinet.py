import queue
import threading
import pyaudio
import numpy as np
import ipywidgets as widgets

import time


class Cabinet:
    def __init__(self, module=None, signal=None):
        self.set_module(module, signal)
        self.output = np.zeros([self.module.framesize], dtype=np.float32)
        self.command_queue = queue.Queue()
        self.setup_gui()
        self.running = False

    def setup_gui(self):
        self.controls = []
        self.start_stop = widgets.Button(
            description='Start',
            disabled=False,
        )
        self.start_stop.on_click(self.on_start_stop)

        self.load = widgets.Label(value="Load: NA")

    def add_float_control(self, label, value, callback, max=1.0, min=0.0):
        label = widgets.Label(label)
        wx = widgets.FloatSlider(
            value=value, min=min, max=max
        )

        def observer(value):
            if value.name == 'value':
                self.call(callback, value.new)

        wx.observe(observer)
        self.add_widget(widgets.HBox([
            label, wx
        ]))

    def add_widget(self, widget):
        self.controls.append(widget)

    def _ipython_display_(self):
        return widgets.VBox(
            [widgets.HBox([
                self.start_stop,
                self.load])] +
            self.controls
        )._ipython_display_()

    def set_module(self, module, signal):
        self.module = module
        self.signal = signal

    def start(self):
        self.running = True
        self.pyaudio = pyaudio.PyAudio()
        t = threading.Thread(target=self.worker, daemon=True)
        t.start()

    def on_start_stop(self, button):
        if self.running:
            text = "Start"
            self.stop()
        else:
            text = "Stop"
            self.start()

        self.start_stop.description = text

    def stop(self):
        self.running = False
        self.pyaudio = None
        self.load.value = "Load: NA"

    def call(self, function, *args, **kwargs):
        self.command_queue.put((function, args, kwargs))

    def worker(self):
        try:
            framesize = self.module.framesize
            terminal_output = open('/dev/stdout', 'w')

            pyad = self.pyaudio
            stream = pyad.open(
                format=pyad.get_format_from_width(4),
                channels=1,
                rate=self.module.samplerate,
                output=True,
                frames_per_buffer=framesize,
            )
            try:
                new_time = start_time = time.time()
                computing = 0
                while self.running:

                    while True:
                        try:
                            f, args, kwargs = self.command_queue.get_nowait()
                            f(*args, **kwargs)
                        except queue.Empty:
                            break

                    self.module.render_frame()
                    end_time = time.time()
                    computing += end_time - new_time
                    stream.write(self.signal.output, framesize)
                    new_time = time.time()

                    elapsed = new_time - start_time
                    if elapsed > 2:
                        load = (computing / elapsed) * 100
                        start_time = new_time
                        computing = 0
                        self.load.value = "Load: %0.2f%%" % (load,)

            finally:
                stream.stop_stream()
                stream.close()
                pyad.terminate()
        except Exception:
            import traceback
            traceback.print_exc(file=terminal_output)


if __name__ == "__main__":
    import synthingie

    mod = synthingie.Module(48000, 1024)
    value = mod.value(440)
    sin = mod.sin(value)
    cabinet = Cabinet(mod, sin)
    cabinet.start()
    time.sleep(1)
    cabinet.call(value.set, 880)
    time.sleep(5)
    cabinet.stop()

import time
import threading
import pyaudio

import numpy as np

import librosa

from bokeh.io import show, output_notebook
import bokeh.models as md
from bokeh.layouts import row, column
from bokeh.plotting import figure, ColumnDataSource

from .core import DEFAULT_FRAMESIZE, DEFAULT_SAMPLERATE, Parameter
from .delay import CircularBuffer


def log(*args):
    # XXX DELETE THIS DIRT
    import os
    os.write(1, (" ".join(str(x) for x in args)).encode("utf8"))


class LiveView:
    samplerate = DEFAULT_SAMPLERATE
    framesize = DEFAULT_FRAMESIZE

    def __init__(self, signal, controls=None):
        self.signal = signal
        signal.configure(self.samplerate, self.framesize)
        self.running = False
        self.paused = False

        self.controls = []
        for step in signal.topological_sort():
            if isinstance(step, Parameter):
                self.controls.append(ParameterController(step))

        if controls is None:
            controls = []
        self.controls.extend(controls)

    def play_clicked(self, what):
        if not self.running:
            self.start()
            self.play_button.label = "Play"
        else:
            if self.paused:
                self.play_button.label = "Play"
                self.paused = False
                self.set_cpu_value(0)
            else:
                self.play_button.label = "Pause"
                self.paused = True

    def stop_clicked(self, what):
        self.running = False
        self.play_button.label = "Play"
        self.paused = False
        self.set_cpu_value(0)

    def _set_cpu_value(self, value):
        self.cpu_data.patch({
            'value': [(0, [value])],
        })

    def set_cpu_value(self, value):
        self.doc.add_next_tick_callback(lambda: self._set_cpu_value(value))
        pass

    def worker(self):
        terminal_output = open('/dev/stdout', 'w')

        try:
            framesize = self.framesize
            samplerate = self.samplerate

            pyad = self.pyaudio
            stream = pyad.open(
                format=pyad.get_format_from_width(4),
                channels=1,
                rate=samplerate,
                output=True,
                frames_per_buffer=framesize,
            )
            try:
                new_time = start_time = time.time()
                computing = 0
                while self.running:
                    if self.paused:
                        time.sleep(1)
                        continue

                    self.signal.render_frame()
                    end_time = time.time()
                    computing += end_time - new_time
                    stream.write(self.signal.output, framesize)
                    new_time = time.time()

                    for control in self.controls:
                        control.tick()

                    elapsed = new_time - start_time
                    if elapsed > 0.25:
                        load = (computing / elapsed) * 100
                        start_time = new_time
                        computing = 0
                        self.set_cpu_value(load)

            finally:
                stream.stop_stream()
                stream.close()
                pyad.terminate()
        except Exception:
            import traceback
            traceback.print_exc(file=terminal_output)

    def __call__(self, doc):
        self.play_button = md.buttons.Button(label="Play")
        self.play_button.on_click(self.play_clicked)

        self.stop_button = md.buttons.Button(label="stop")
        self.stop_button.on_click(self.stop_clicked)

        self.cpu_data = ColumnDataSource(dict(label=['CPU'], value=[0]))
        cpu = figure(y_range=self.cpu_data.data['label'], plot_height=50, plot_width=300,
                     toolbar_location=None, tools="")
        cpu.hbar(y="label", left="value", height=1, source=self.cpu_data)
        cpu.ygrid.grid_line_color = None
        cpu.x_range.start = 0
        cpu.x_range.end = 100
        cpu.min_border_right = 10
        self.cpu_meter = cpu

        for control in self.controls:
            control.configure(doc, self.samplerate, self.framesize)

        doc.add_root(column(*(
            [row(self.play_button, self.stop_button, self.cpu_meter)] +
            [p.get_layout() for p in self.controls]
        )))

        # I am sure i should not keep this around as bokeh might want to
        # have many docs per class. Let's wait and see when this breaks.
        self.doc = doc
        self.start()

    def start(self):
        self.running = True
        self.set_cpu_value(0)
        self.pyaudio = pyaudio.PyAudio()
        t = threading.Thread(target=self.worker, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        self.pyaudio = None
        self.set_cpu_value(0)

    def _ipython_display_(self):
        output_notebook()
        show(self)


class LiveControl:

    def configure(self, doc, samplerate, framesize):
        "Initialize."

    def get_layout(self):
        "Return a layout to render."

    def tick(self):
        "Called every frame to read the signals. "


class ParameterController(LiveControl):
    def __init__(self, parameter):
        self.parameter = parameter

    def on_value_change(self, attr, old, new):
        self.parameter.set(new)

    def get_layout(self):
        step = self.parameter.step
        if step is None:
            step = (self.parameter.max - self.parameter.min) / 1000

        slider = md.Slider(
            start=self.parameter.min, end=self.parameter.max,
            title=self.parameter.name, value=self.parameter.value,
            step=step
        )
        slider.on_change("value", self.on_value_change)
        return slider


class Scope(LiveControl):
    # XXX make this class not so ugly
    point_count = 1024
    framerate = 10

    def __init__(self, signal):
        self.signal = signal

    def configure(self, doc, samplerate, framesize):
        self.samplerate = samplerate
        self.framesize = framesize

        self.doc = doc
        self.elapsed_samples = 0
        self.buffer_size = samplerate
        self.buffer = CircularBuffer(self.buffer_size)

        self.trigger_on = False
        self.triggered_at = None
        self.raise_or_fall_value = 0

        self.trigger_points = np.ones([self.point_count])
        self.display_points = np.ones([self.point_count])
        self.display_ticks = np.zeros([self.point_count])

        self.source = ColumnDataSource(data=dict(
            x=self.display_ticks,
            y=self.display_points,
            trigger=self.trigger_points,
        ))

        self.plot = figure(
            plot_height=400, plot_width=400,
            tools="crosshair,pan,reset,save,wheel_zoom",
            x_range=[0, 10],
            y_range=[-2.5, 2.5]
        )

        self.vline = md.Span(
            location=5, dimension='height', line_color='red', line_width=1,
            line_dash='dotted', line_alpha=0
        )

        self.set_time_width(10)
        self.set_trigger_value(2)

    def set_time_width(self, width_ms):
        self.width_ms = width_ms
        self.display_ticks[:] = np.linspace(0, width_ms, self.point_count)
        self.source.data['x'] = self.display_ticks
        self.plot.x_range.end = self.width_ms
        self.vline.location = width_ms / 2

    def on_width_moved(self, attr, old, new):
        self.set_time_width(new)

    def set_trigger_value(self, value):
        self.trigger_value = value
        self.trigger_points[:] = value
        self.source.data['trigger'] = self.trigger_points

    def raise_or_fall_clicked(self, what, old, new):
        self.raise_or_fall_value = new

    def tick(self):
        # XXX trigger position

        redraw = False
        self.buffer.add(self.signal.output)
        self.elapsed_samples += self.framesize

        if self.trigger_on:
            if self.triggered_at is None:
                # wait for a trigger, register position
                data = self.signal.output

                if self.raise_or_fall_value == 0:
                    triggers = np.flatnonzero(
                            (data[:-1] < self.trigger_value) &
                            (data[1:] > self.trigger_value)
                    ) + 1
                else:
                    triggers = np.flatnonzero(
                            (data[:-1] > self.trigger_value) &
                            (data[1:] < self.trigger_value)
                    ) + 1

                if len(triggers):
                    self.triggered_at = self.framesize - triggers[0]
                    self.elapsed_samples = self.triggered_at
                    self.trigger_drawn = False
                elif self.elapsed_samples > self.samplerate / self.framerate:
                    redraw = True

            else:
                self.triggered_at += self.framesize

            # if we have collected enough samples past trigger, redraw
            # how many samples is half a view?
            if not self.trigger_drawn:
                frames_required = int(self.width_ms / 1000 * self.samplerate / 2)
                if self.triggered_at is not None and self.triggered_at > frames_required:
                    # send update
                    new_points = self.buffer.index(
                        self.buffer.buffer.shape[0] - np.linspace(
                            self.width_ms * self.samplerate / 1000 - 1, 0,
                            self.point_count, dtype=int
                        ) - self.triggered_at + frames_required
                    )

                    def update():
                        self.vline.line_alpha = 0.6
                        self.source.data['y'] = new_points
                    self.doc.add_next_tick_callback(update)
                    self.trigger_drawn = True

            if self.triggered_at is not None:
                if self.trigger_drawn and self.elapsed_samples > self.samplerate / self.framerate:
                    self.triggered_at = None
        else:
            if self.elapsed_samples > self.samplerate / self.framerate:
                redraw = True

        if redraw:
            self.elapsed_samples = 0
            new_points = self.buffer.index(
                self.buffer.buffer.shape[0] - np.linspace(
                    self.width_ms * self.samplerate / 1000 - 1, 0,
                    self.point_count, dtype=int
                )
            )

            def update():
                self.vline.line_alpha = 0.0
                self.source.data['y'] = new_points
            self.doc.add_next_tick_callback(update)

    def trigger_clicked(self, what):
        self.triggered_at = None
        self.trigger_drawn = False

        if self.trigger_on:
            self.trigger_plot.glyph.line_alpha = 0
            self.trigger_on = False
        else:
            self.trigger_plot.glyph.line_alpha = 0.6
            self.trigger_on = True

    def on_trigger_moved(self, attr, old, new):
        self.set_trigger_value(new)

    def on_scale_moved(self, attr, old, new):
        self.plot.y_range.start = -new
        self.plot.y_range.end = new

    def get_layout(self):
        # Set up plot
        plot = self.plot

        plot.line('x', 'y', source=self.source, line_width=3, line_alpha=0.6)

        plot.renderers.append(self.vline)

        self.trigger_plot = plot.line('x', 'trigger', source=self.source, line_width=3, line_alpha=0.6)
        self.trigger_plot.glyph.line_alpha = 0
        self.trigger_plot.glyph.line_color = "#15bd42"
        self.trigger_plot.glyph.line_width = 1

        trigger_slider = md.Slider(
            start=-2.5, end=2.5,
            value=2, step=0.01,
            title="Trigger Value"
        )
        trigger_slider.on_change("value", self.on_trigger_moved)

        width_slider = md.Slider(
            start=1, end=1000,
            value=10, step=1,
            title='Width (ms)'
        )
        width_slider.on_change("value", self.on_width_moved)

        scale_slider = md.Slider(
            start=0.1, end=10,
            value=1, step=0.01,
            title='Scale'
        )
        scale_slider.on_change("value", self.on_scale_moved)

        self.trigger_button = md.buttons.Toggle(label="Trigger")
        self.trigger_button.on_click(self.trigger_clicked)

        raise_or_fall = md.RadioButtonGroup(
            labels=["Raising", "Falling"], active=self.raise_or_fall_value)
        raise_or_fall.on_change('active', self.raise_or_fall_clicked)

        scope = row(
            plot,
            column(
                row(
                    self.trigger_button,
                    raise_or_fall,
                ),
                trigger_slider,
                width_slider,
                scale_slider,
            ),
        )
        self.layour = scope

        return scope


class Analyzer(LiveControl):
    point_count = 1024
    framerate = 10

    def __init__(self, signal):
        self.signal = signal

    def configure(self, doc, samplerate, framesize):
        self.samplerate = samplerate
        self.framesize = framesize

        self.doc = doc
        self.elapsed_samples = 0
        self.buffer_size = int(samplerate)
        self.buffer = CircularBuffer(self.buffer_size)

        self.display_ticks = librosa.core.fft_frequencies(sr=samplerate, n_fft=4096)
        self.display_points = np.ones([self.display_ticks.shape[0]])

        self.source = ColumnDataSource(data=dict(
            x=self.display_ticks,
            y=self.display_points,
        ))

    def tick(self):
        self.buffer.add(self.signal.output)
        self.elapsed_samples += self.framesize

        if self.elapsed_samples > self.samplerate / self.framerate:
            self.elapsed_samples = 0
            points = self.buffer.index(np.arange(self.buffer_size))

            ffted = librosa.core.stft(points[-4096:], n_fft=4096, center=False)
            ffted = ffted[:, -1]
            new_points = np.abs(ffted)

            def update():
                self.source.data['y'] = new_points
            self.doc.add_next_tick_callback(update)

    def get_layout(self):
        # Set up plot
        plot = figure(
            plot_height=400, plot_width=800,
            tools="crosshair,pan,reset,save,wheel_zoom,hover",
            x_axis_type="log",
            tooltips=[("Hz", "$x")],
        )
        plot.xaxis.formatter = md.BasicTickFormatter(use_scientific=False)
        plot.line('x', 'y', source=self.source, line_width=1, line_alpha=0.6)

        return plot

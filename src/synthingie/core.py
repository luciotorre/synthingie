from functools import wraps
import time
from typing import Union

import numpy as np

# and IPython.display for audio output
import IPython.display as ipd

import matplotlib.pyplot as plt
import matplotlib.style as ms

# Librosa for audio
import librosa
import librosa.display

from scipy.io.wavfile import write, read

from .score import Score

ms.use('seaborn-muted')


class Audio:
    def __init__(self, samplerate, samples):
        self.samplerate = samplerate
        self.samples = samples

    def __getitem__(self, args):
        if isinstance(args, slice):
            return self.__class__(self.samplerate, self.samples[args])
        else:
            return self.samples[args]

    def display(self, limit_samples=None):
        # to make things compatible, transpose into librosa format and make mono
        samples = librosa.to_mono(self.samples.T)

        # from librosa's tutorial
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(samples, sr=self.samplerate, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)

        # Make a new figure
        plt.figure(figsize=(12, 4))

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=self.samplerate, x_axis='time', y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram')

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()
        plt.figure(figsize=(12, 4))

        self.display_signal(limit_samples)

        # FIXME: audio played is mono! should be stereo if source was stereo
        ipd.display(ipd.Audio(samples, rate=self.samplerate))

    def display_signal(self, limit_samples=None):
        samples = librosa.to_mono(self.samples.T)

        if limit_samples is not None:
            plt.plot(samples[:limit_samples])
        else:
            plt.plot(samples)

    def save(self, filename):
        assert filename.lower().endswith(".wav")

        # convert to float32 because aplay, mplayer and friends cant handle 64bit wavs
        write(filename, self.samplerate, self.samples.astype(np.float32))

    @classmethod
    def load(cls, filename):
        samplerate, samples = read(filename)
        if samples.dtype != np.float32:
            max_value = np.max(np.abs(samples))
            samples = samples.astype(np.float32) / max_value
        return cls(samplerate, samples)


class Signal:
    """Base op, does nothing."""

    dtype = np.float32

    def bind(self, module):
        self.module = module
        self.samplerate = module.samplerate
        self.framesize = module.framesize

        # we pre generate our output buffer to not allocate on runtime
        self.output = np.zeros(self.framesize, dtype=self.dtype)

    def __call__(self):
        """Called to generate a new frame for this signal."""


SignalTypes = Union[Signal, float, int]


def signal_value(signal_or_value: SignalTypes):
    if isinstance(signal_or_value, Signal):
        return signal_or_value.output
    else:
        return signal_or_value  # is value


class Module:
    def __init__(self, samplerate: int, framesize: int):
        assert isinstance(samplerate, int), "int required"
        assert isinstance(framesize, int), "int required"
        self.module = self
        self.samplerate = samplerate
        self.framesize = framesize
        self._steps = []

    def profile(self, duration_s, score=None):
        # How many frames do we need?
        num_frames = int(duration_s * self.samplerate / self.framesize) + 1
        output = np.zeros(num_frames * self.framesize)

        frame_times = []
        step_times = [[] for s in self._steps]
        event_times = []

        # Setup score
        frame_duration_s = self.framesize / self.samplerate
        score_runner = None
        if score is not None:
            score_runner = score.run()

        for i in range(num_frames):
            start = time.time()
            if score_runner is not None:
                start_time = time.perf_counter_ns()
                score_runner.advance(frame_duration_s)
                event_times.append(time.perf_counter_ns() - start_time)

            for j, step in enumerate(self._steps):
                start_time = time.perf_counter_ns()
                step()
                step_times[j].append(time.perf_counter_ns() - start_time)

            frame_times.append(time.time() - start)
            output[i * self.framesize:(i + 1) * self.framesize] = step.output

        frame_duration = self.framesize / self.samplerate
        print("{} runs of {:.2f}ms framesize each.".format(
            len(frame_times), 1000 * frame_duration
        ))
        print("Processing duration {:.2f}ms avg, {:.2f}ms median, {:.2f}ms max, {:.2f}ms min.".format(
            sum(frame_times) / len(frame_times) * 1000,
            np.median(frame_times) * 1000,
            max(frame_times) * 1000,
            min(frame_times) * 1000
        ))
        print("Processing duration {:.2f}% avg, {:.2f}% median, {:.2f}% max, {:.2f}% min of sample duration.".format(
            sum(frame_times) / len(frame_times) / frame_duration * 100,
            np.median(frame_times) / frame_duration * 100,
            max(frame_times) / frame_duration * 100,
            min(frame_times) / frame_duration * 100
        ))

        def print_header():
            print_footer()
            print("| {:20s} | {:11s} | {:11s} | {:11s} | {:11s} |".format(
                "", "avg", "median", "max", "min"
            ))
            print_footer()

        def print_footer():
            print("+" + "-" * 22 + "+-------------" * 4 + "+")

        def print_stats(label, times):
            print("| {:20s} | {:9.2f}µs | {:9.2f}µs | {:9.2f}µs | {:9.2f}µs |".format(
                label,
                sum(times) / len(times) / 1000,
                np.median(times) / 1000,
                max(times) / 1000,
                min(times) / 1000
            ))
        print()
        print_header()
        for step, times in zip(self._steps, step_times):
            print_stats(str(step.__class__.__name__), times)
        print_footer()

    def render_frame(self):
        for step in self._steps:
            step()

    def render(
            self, target: Signal, duration_s: float,
            score: Score = None, profile: bool = False) -> Audio:
        times = []

        # How many frames do we need?
        num_frames = int(duration_s * self.samplerate / self.framesize) + 1
        output = np.zeros(num_frames * self.framesize)

        # Setup score
        frame_duration_s = self.framesize / self.samplerate
        score_runner = None
        if score is not None:
            score_runner = score.run()

        for i in range(num_frames):
            start = time.time()
            if score_runner is not None:
                score_runner.advance(frame_duration_s)
            self.render_frame()
            times.append(time.time() - start)
            output[i * self.framesize:(i + 1) * self.framesize] = target.output

        if profile:
            frame_duration = self.framesize / self.samplerate
            print("{} runs of {:.2f}ms sample duration each.".format(
                len(times), 1000 * frame_duration
            ))
            print("Processing duration {:.2f}ms avg, {:.2f}ms max, {:.2f}ms min.".format(
                sum(times) / len(times) * 1000, max(times) * 1000, min(times) * 1000
            ))
            print("Processing duration {:.2f}% avg, {:.2f}% max, {:.2f}% min of sample duration.".format(
                sum(times) / len(times) / frame_duration * 100,
                max(times) / frame_duration * 100,
                min(times) / frame_duration * 100
            ))
        return Audio(self.samplerate, output[:int(duration_s * self.samplerate)])

    def add_step(self, signal):
        signal.bind(self)
        self._steps.append(signal)


def register(base_class, method_name, overwrite=False):
    accepted = [Signal, Module]
    if base_class not in accepted:
        raise ValueError("Invalid base class: %s, expecting one of %s" % (
                base_class,
                accepted
            ))

    if not overwrite and hasattr(base_class, method_name):
        raise NameError("Class %s: name '%s' already in use" % (base_class, method_name))

    def decorator(method_class):
        @wraps(method_class)
        def wrapper(self, *args, **kwargs):
            if isinstance(self, Signal):
                operation = method_class(self, *args, **kwargs)
            else:
                operation = method_class(*args, **kwargs)
            self.module.add_step(operation)
            return operation
        setattr(base_class, method_name, wrapper)
        return method_class
    return decorator


@register(Module, "value")
class Value(Signal):
    def __init__(self, value: float):
        self.set(value)

    def set(self, value):
        accepted = (float, int)
        if not isinstance(value, accepted):
            raise ValueError("Value '%s' not in accepted types: %s" % (
                value, accepted
            ))
        self._value = float(value)

    def __call__(self):
        self.output.fill(self._value)

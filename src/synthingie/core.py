__all__ = ["Signal", "Audio", "Value", "Parameter", "signal", "control_method"]

from functools import wraps
import time
from dataclasses import dataclass, fields
from copy import copy

import numpy as np

# and IPython.display for audio output
import IPython.display as ipd

import matplotlib.pyplot as plt

# Librosa for audio
import librosa
import librosa.display

from scipy.io.wavfile import write, read

from .score import Score


DEFAULT_SAMPLERATE = 48000
DEFAULT_FRAMESIZE = int(DEFAULT_SAMPLERATE / 60)  # 60hz/fps rendering


class Audio:
    def __init__(self, samplerate, samples):
        self.samplerate = samplerate
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, args):
        if isinstance(args, slice):
            return self.__class__(self.samplerate, self.samples[args])
        else:
            return self.samples[args]

    def rms(self):
        return np.sqrt(np.sum(self.samples ** 2) / self.samples.shape[0])

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

    _ipython_display_ = display

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


@dataclass(order=False, eq=True)
class Signal:
    """Base op, does nothing."""

    dtype = np.float32

    def __post_init__(self):
        """Make sure all signal values are boxed."""
        for field in fields(self):
            if issubclass(field.type, Signal):
                if not isinstance(getattr(self, field.name), Signal):
                    value = getattr(self, field.name)
                    if not isinstance(value, (int, float)):
                        raise ValueError("Expected signal, int or float on field '%s', got: '%s'." % (
                            field.name, value
                        ))
                    setattr(self, field.name, Value(value))

        # Still not ready to produce sound
        self.is_configured = False

        # all control actions get recorded here to be run in the synthesis loop
        # This lets you call all control methods from a different thread and call them before
        # configuring the patch
        self.pending_control = []

    def __hash__(self):
        return hash(id(self))

    # # Patch definition methods
    # These methods work on DAG of operations and only on the dataclass defined
    # values. For these methods, an instance is an immutable value.

    def to_json(self):
        # XXX
        pass

    @classmethod
    def from_json(self, string: str):
        # XXX
        pass

    def sources(self):
        """Return all the signal instances that input into this one."""
        for field in fields(self):
            if issubclass(field.type, Signal):
                yield getattr(self, field.name)

    def copy(self):
        """Return a copy of this patch, useful to start using the mutation methods."""

        return self.__copy__()

    def __copy__(self):
        kwargs = {}
        for field in fields(self):
            kwargs[field.name] = copy(getattr(self, field.name))

        instance = self.__class__(**kwargs)
        return instance

    def topological_sort(self):
        """Return the nodes in this patch in topological order.

        Each operation is a new node in the DAG of operations required to build this signal.
        This method will return the list of nodes/operations sorted in a way in wich not
        operation is listed before the operations in its inputs.
        """
        result = []
        done = set()
        open_nodes = set()

        def visit(node):
            if node in done:
                return
            if node in open_nodes:
                raise ValueError("Signal path is not a DAG, cycle detected")

            open_nodes.add(node)

            for source in node.sources():
                visit(source)

            open_nodes.remove(node)
            done.add(node)
            result.append(node)

        visit(self)

        return result

    def list_steps(self):
        """Return the ordered steps plus a printable representation of each one."""
        steps = self.topological_sort()

        positions = dict((s, i) for i, s in enumerate(steps))

        def repr_step(step):
            args = []
            for field in fields(step):
                value = getattr(step, field.name)
                if issubclass(field.type, Signal):
                    args.append("%s=%s.%s" % (field.name, positions[value], value.__class__.__name__))
                else:
                    args.append("%s=%s" % (field.name, value))

            return "%s.%s(%s)" % (
                positions[step],
                step.__class__.__name__,
                ", ".join(args)
            )

        return steps, [repr_step(step) for step in steps]

    def pipe(self, other, *args, **kwargs):
        """Pipe the output of this signal into the input of another one.

        The `other` signal will be instantiated with (self, *args, **kwargs).
        This works with signals that allow the "pipe protocol", ie, the first argument is the
        input.
        """
        return other(self, *args, **kwargs)

    # # Processing methods
    # These methods actually process data to generate signals. These methods might
    # keep state between calls.

    def configure(self, samplerate=DEFAULT_SAMPLERATE, framesize=DEFAULT_FRAMESIZE):
        """Set rendering parameters and create output buffers for patch.

        In most cases convenience methods will call this for you.
        """

        steps = self.topological_sort()

        for step in steps:
            step._pre_setup(samplerate, framesize)
            step.setup()

    def _pre_setup(self, samplerate, framesize):
        self.samplerate = samplerate
        self.framesize = framesize

        # we pre generate our output buffer to not allocate on runtime
        self.output = np.zeros(self.framesize, dtype=self.dtype)
        self.is_configured = True

    def setup(self):
        """__init__ method replacement. Override this.

        This is a dataclass, so we leave the automatic __init__ method alone. That ensures that every instance
        can represent a patch in any context, leaving the actual initialization of the synthesis data to the
        moment when we already know the framesize and samplerate.
        """
        pass

    def __call__(self):
        """Store one frame of data in self.output. Override this."""
        pass

    # # Utility methods
    # These methods cross the boundary between the two other type. Practicality beats purity.

    def render_frame(self, steps=None):
        """Render one frame of output.

        Dont use this method in performance sensitive code without providing the steps first.
        See example in `.render()`
        """
        if steps is None:
            steps = self.topological_sort()

        if not self.is_configured:
            self.configure(DEFAULT_SAMPLERATE, DEFAULT_FRAMESIZE)

        for step in steps:
            for control_method, args, kwargs in step.pending_control:
                control_method(step, *args, **kwargs)
            step.pending_control = []
            step()

    def render(self,
               duration_s: float, score: Score = None,
               samplerate: int = DEFAULT_SAMPLERATE,
               framesize: int = DEFAULT_FRAMESIZE) -> Audio:

        steps = self.topological_sort()

        if not self.is_configured:
            self.configure(DEFAULT_SAMPLERATE, DEFAULT_FRAMESIZE)

        # How many frames do we need?
        num_frames = int(duration_s * self.samplerate / self.framesize) + 1
        output = np.zeros(num_frames * self.framesize)

        # Setup score
        frame_duration_s = self.framesize / self.samplerate
        score_runner = None
        if score is not None:
            score_runner = score.run()

        for i in range(num_frames):
            if score_runner is not None:
                score_runner.advance(frame_duration_s)
            self.render_frame(steps)
            output[i * self.framesize:(i + 1) * self.framesize] = self.output

        return Audio(self.samplerate, output[:int(duration_s * self.samplerate)])

    def profile(self, duration_s: float, score: Score = None):
        if not self.is_configured:
            self.configure(DEFAULT_SAMPLERATE, DEFAULT_FRAMESIZE)

        # How many frames do we need?
        num_frames = int(duration_s * self.samplerate / self.framesize) + 1
        output = np.zeros(self.framesize)

        steps, labels = self.list_steps()

        frame_times = []
        step_times = [[] for s in steps]
        step_control_times = [[] for s in steps]
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

            for j, step in enumerate(steps):
                start_time = time.perf_counter_ns()
                for control_method, args, kwargs in step.pending_control:
                    control_method(step, *args, **kwargs)
                step.pending_control = []
                step_control_times[j].append(time.perf_counter_ns() - start_time)

                start_time = time.perf_counter_ns()
                step()
                step_times[j].append(time.perf_counter_ns() - start_time)

            output[:] = step.output
            frame_times.append(time.time() - start)

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
            print("| {:80s} | {:11s} | {:11s} | {:11s} | {:11s} |".format(
                "", "avg", "median", "max", "min"
            ))
            print_footer()

        def print_footer():
            print("+" + "-" * 82 + "+-------------" * 4 + "+")

        def print_stats(label, times):
            print("| {:80s} | {:9.2f}µs | {:9.2f}µs | {:9.2f}µs | {:9.2f}µs |".format(
                label[:80],
                sum(times) / len(times) / 1000,
                np.median(times) / 1000,
                max(times) / 1000,
                min(times) / 1000
            ))
        print()
        print_header()
        for label, times in zip(labels, step_times):
            print_stats(label, times)
        print_footer()

    def live(self, controls=None):
        from .live import LiveView

        return LiveView(self, controls)


def control_method(method):
    """Decorate method used to control signals."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.pending_control.append((method, args, kwargs))

    return wrapper


def signal(arg):
    """Class decorator for signals."""
    if isinstance(arg, type):
        klass = arg

        new_class = dataclass(order=False, eq=True)(klass)
        new_class.__hash__ = lambda self: hash(id(self))
        return new_class

    elif isinstance(arg, str):
        method_name = arg

        def decorator(base_class):
            klass = signal(base_class)

            @wraps(klass)
            def wrapper(self, *args, **kwargs):
                operation = klass(self, *args, **kwargs)
                return operation

            setattr(Signal, method_name, wrapper)
            return klass
        return decorator


@signal
class Value(Signal):
    """A wrapper for scalar values."""
    value: float

    def __call__(self):
        # XXX: This should be a scalar and use broadcasting
        self.output.fill(self.value)

    @control_method
    def set(self, value):
        self.value = value


@signal
class Parameter(Value):
    """A parameter."""
    name: str
    min: float = 0
    max: float = 1
    step: float = None

    def __call__(self):
        self.output.fill(self.value)

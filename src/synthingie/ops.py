import numpy as np

from .core import Signal, SignalTypes, register


class Plus(Signal):
    """Add a signal and a signal or value.

    >>> one = module.value(1)
    >>> two = module.value(2)
    >>> result = 3 + one + two + 3

    >>> module.render_frames()
    >>> assert np.all(result.output == 9.0)
    """
    def init(self, signal: Signal, other: SignalTypes):
        self.left = signal.output
        self.right = self.module.as_signal(other).output

    def __call__(self):
        np.add(self.left, self.right, out=self.output)


register(Signal, "__add__")(Plus)
register(Signal, "__radd__")(Plus)


@register(Signal, "__sub__")
class Minus(Signal):
    """Subtract a signal or value from a signal.

    >>> one = module.value(1)
    >>> result = one - one - 1

    >>> module.render_frames()
    >>> assert np.all(result.output == -1.0)
    """
    def init(self, signal: Signal, other: SignalTypes):
        """Subtract a signal or value from a signal."""
        self.left = signal.output
        self.right = self.module.as_signal(other).output

    def __call__(self):
        np.subtract(self.left, self.right, out=self.output)


@register(Signal, "__rsub__")
class RMinus(Signal):
    """Subtract a signal from a value.

    >>> one = module.value(1)
    >>> result = 2 - one

    >>> module.render_frames()
    >>> assert np.all(result.output == 1.0)
    """
    def init(self, signal: Signal, other: SignalTypes):

        self.right = signal.output
        self.left = self.module.as_signal(other).output

    def __call__(self):
        np.subtract(self.left, self.right, out=self.output)


class Mul(Signal):
    """Multiply a signal and a signal or value.

    >>> two = module.value(2)
    >>> three = module.value(3)
    >>> result = 1 * two * three * 4

    >>> module.render_frames()
    >>> assert np.all(result.output == 24.0)
    """

    def init(self, signal: Signal, other: SignalTypes):

        self.left = signal.output
        self.right = self.module.as_signal(other).output

    def __call__(self):
        np.multiply(self.left, self.right, out=self.output)


register(Signal, "__mul__")(Mul)
register(Signal, "__rmul__")(Mul)

# @register("clip")
# class ClipOp(Op):
#     """Clip signal between extremes

#     >>> run(48000, 1024, "[ 1 2 3 ] 1.5 2.5 clip")
#     [array([1.5, 2. , 2.5])]
#     """
#     def __call__(self, stack):
#         upper = stack.pop()
#         lower = stack.pop()
#         data = stack.pop()

#         stack.push(np.clip(data, lower, upper))

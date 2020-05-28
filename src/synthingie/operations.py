import numpy as np

from .core import Signal, SignalTypes, register, signal_value


class Operation(Signal):
    def __init__(self, signal: Signal, other: SignalTypes):
        self.left = signal.output
        self.right = signal_value(other)


class Plus(Operation):
    """Add a signal and a signal or value.

    >>> one = module.value(1)
    >>> two = module.value(2)
    >>> result = 3 + one + two + 3

    >>> module.render_frame()
    >>> assert np.all(result.output == 9.0)
    """

    def __call__(self):
        np.add(self.left, self.right, out=self.output)


register(Signal, "__add__")(Plus)
register(Signal, "__radd__")(Plus)


@register(Signal, "__sub__")
class Minus(Operation):
    """Subtract a signal or value from a signal.

    >>> one = module.value(1)
    >>> result = one - one - 1

    >>> module.render_frame()
    >>> assert np.all(result.output == -1.0)
    """

    def __call__(self):
        np.subtract(self.left, self.right, out=self.output)


@register(Signal, "__rsub__")
class RMinus(Operation):
    """Subtract a signal from a value.

    >>> one = module.value(1)
    >>> result = 2 - one

    >>> module.render_frame()
    >>> assert np.all(result.output == 1.0)
    """

    def __call__(self):
        np.subtract(self.right, self.left, out=self.output)


class Mul(Operation):
    """Multiply a signal and a signal or value.

    >>> two = module.value(2)
    >>> three = module.value(3)
    >>> result = 1 * two * three * 4

    >>> module.render_frame()
    >>> assert np.all(result.output == 24.0)
    """

    def init(self, signal: Signal, other: SignalTypes):

        self.left = signal.output
        self.right = self.module.as_signal(other).output

    def __call__(self):
        np.multiply(self.left, self.right, out=self.output)


register(Signal, "__mul__")(Mul)
register(Signal, "__rmul__")(Mul)


@register(Signal, "__mod__")
class Mod(Operation):
    """Calculate the modulus of a signal.

    >>> two = module.value(2)
    >>> three = module.value(3)
    >>> result = three % two

    >>> module.render_frame()
    >>> assert np.all(result.output == 1)
    """
    def __call__(self):
        np.mod(self.left, self.right, out=self.output)


@register(Signal, "__lt__", overwrite=True)
class LT(Operation):
    """Calculate the modulus of a signal.

    >>> two = module.value(2)
    >>> three = module.value(3)
    >>> result = three < two

    >>> module.render_frame()
    >>> assert np.all(result.output == 0)
    """

    def __call__(self):
        np.less(self.left, self.right, out=self.output)


@register(Signal, "__gt__", overwrite=True)
class GT(Operation):
    """Calculate the modulus of a signal.

    >>> two = module.value(2)
    >>> three = module.value(3)
    >>> result = three > two

    >>> module.render_frame()
    >>> assert np.all(result.output == 1)
    """

    def __call__(self):
        np.greater(self.left, self.right, out=self.output)


@register(Signal, "__abs__")
class Abs(Signal):
    """Subtract a signal from a value.

    >>> one = module.value(-1)
    >>> result = abs(one)

    >>> module.render_frame()
    >>> assert np.all(result.output == 1.0)
    """

    def __init__(self, signal: Signal):
        self.signal = signal_value(signal)

    def __call__(self):
        np.abs(self.signal, out=self.output)


@register(Signal, "__pow__")
class Pow(Operation):
    """Calculate the modulus of a signal.

    >>> two = module.value(2)
    >>> three = module.value(3)
    >>> result = three ** two

    >>> module.render_frame()
    >>> assert np.all(result.output == 3 ** 2)
    """

    def __call__(self):
        np.power(self.left, self.right, out=self.output)


@register(Signal, "__rpow__")
class RPow(Operation):
    """Calculate the modulus of a signal.

    >>> two = module.value(2)
    >>> result = 2 ** two

    >>> module.render_frame()
    >>> assert np.all(result.output == 2 ** 2)
    """

    def __call__(self):
        np.power(self.right, self.left, out=self.output)

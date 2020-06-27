import numpy as np

from .core import Signal, signal


@signal
class Operation(Signal):
    left: Signal
    right: Signal


class Plus(Operation):
    """Add a signal and a signal or value.

    >>> one = Value(1)
    >>> two = Value(2)
    >>> result = 3 + one + two + 3

    >>> result.render_frame()
    >>> assert np.all(result.output == 9.0)
    """

    def __call__(self):
        np.add(self.left.output, self.right.output, out=self.output)


signal("__add__")(Plus)
signal("__radd__")(Plus)


@signal("__sub__")
class Minus(Operation):
    """Subtract a signal or value from a signal.

    >>> one = Value(1)
    >>> result = one - one - 1

    >>> result.render_frame()
    >>> assert np.all(result.output == -1.0)
    """

    def __call__(self):
        np.subtract(self.left.output, self.right.output, out=self.output)


@signal("__rsub__")
class RMinus(Operation):
    """Subtract a signal from a value.

    >>> one = Value(1)
    >>> result = 2 - one

    >>> result.render_frame()
    >>> assert np.all(result.output == 1.0)
    """

    def __call__(self):
        np.subtract(self.right.output, self.left.output, out=self.output)


class Mul(Operation):
    """Multiply a signal and a signal or value.

    >>> two = Value(2)
    >>> three = Value(3)
    >>> result = 1 * two * three * 4

    >>> result.render_frame()
    >>> assert np.all(result.output == 24.0)
    """

    def __call__(self):
        np.multiply(self.left.output, self.right.output, out=self.output)


signal("__mul__")(Mul)
signal("__rmul__")(Mul)


@signal("__mod__")
class Mod(Operation):
    """Calculate the modulus of a signal.

    >>> two = Value(2)
    >>> three = Value(3)
    >>> result = three % two

    >>> result.render_frame()
    >>> assert np.all(result.output == 1)
    """
    def __call__(self):
        np.mod(self.left.output, self.right.output, out=self.output)


@signal("__lt__")
class LT(Operation):
    """Calculate the modulus of a signal.

    >>> two = Value(2)
    >>> three = Value(3)
    >>> result = three < two

    >>> result.render_frame()
    >>> assert np.all(result.output == 0)
    """

    def __call__(self):
        np.less(self.left.output, self.right.output, out=self.output)


@signal("__gt__")
class GT(Operation):
    """Calculate the modulus of a signal.

    >>> two = Value(2)
    >>> three = Value(3)
    >>> result = three > two

    >>> result.render_frame()
    >>> assert np.all(result.output == 1)
    """

    def __call__(self):
        np.greater(self.left.output, self.right.output, out=self.output)


@signal("__abs__")
class Abs(Signal):
    """Subtract a signal from a value.

    >>> one = Value(-1)
    >>> result = abs(one)

    >>> result.render_frame()
    >>> assert np.all(result.output == 1.0)
    """

    signal: Signal

    def __call__(self):
        np.abs(self.signal.output, out=self.output)


@signal("__pow__")
class Pow(Operation):
    """Calculate the modulus of a signal.

    >>> two = Value(2)
    >>> three = Value(3)
    >>> result = three ** two

    >>> result.render_frame()
    >>> assert np.all(result.output == 3 ** 2)
    """

    def __call__(self):
        np.power(self.left.output, self.right.output, out=self.output)


@signal("__rpow__")
class RPow(Operation):
    """Calculate the modulus of a signal.

    >>> two = Value(2)
    >>> result = 2 ** two

    >>> result.render_frame()
    >>> assert np.all(result.output == 2 ** 2)
    """

    def __call__(self):
        np.power(self.right.output, self.left.output, out=self.output)

"""Test suite for the pow method of the Scalar object."""

from math import isclose, log
from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet

def test_pow_with_label() -> None:
    """Test the pow method of the Scalar object with a label."""
    x = Scalar(2.0, requires_grad=True)
    y = Scalar(3.0, requires_grad=True)
    z = x.pow(y, label="z")
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 8.0
    assert z._op == Operation.EXPONENTIATION
    assert z._prev == OrderedSet([x, y])
    assert z.label == "z"
    assert isclose(x._grad, (x.data**(y.data-1.0)) * y.data)
    assert isclose(y._grad, (x.data**y.data) * log(x.data))

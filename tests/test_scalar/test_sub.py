"""Test suit for the sub method of the Scalar object."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet

def test_sub_with_label() -> None:
    """Test the sub method with a label."""
    x = Scalar(1.0, requires_grad=True, label="x")
    y = Scalar(2.0, requires_grad=True, label="y")
    z = x.sub(y, label="z")
    z._grad = 1.0
    z._backward_fn()

    assert z.data == -1.0
    assert z._op == Operation.SUBTRACTION
    assert z._prev == OrderedSet([x, y])
    assert x._grad == 1.0
    assert y._grad == -1.0
    assert z.label == "z"
    assert x.label == "x"
    assert y.label == "y"

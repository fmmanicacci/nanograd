"""Test suite for the __rfloordiv__ method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__rfloordiv__int() -> None:
    """Test the __rfloordiv__ method of the Scalar object with an int."""
    x = Scalar(1.143498, requires_grad=True)
    z = 2 // x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 1.0
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {y, x}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test__rfloordiv__float() -> None:
    """Test the __rfloordiv__ method of the Scalar object with a float."""
    x = Scalar(1.143498, requires_grad=True)
    z = 2.0 // x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == 1.0
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {y, x}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test__rfloordiv__unsupported() -> None:
    """Test the __rfloordiv__ method of the Scalar object with an unsupported type."""
    x = Scalar(1.143498, requires_grad=True)
    with raises(TypeError):
        "2.0" // x

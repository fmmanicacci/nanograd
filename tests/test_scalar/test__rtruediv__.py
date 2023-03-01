"""Test suit for the __rtruediv__ method of the Scalar object."""

from math import isclose
from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__rtruediv__int() -> None:
    """Test the __rtruediv__ method with an argument of type int."""
    y = Scalar(2.0, requires_grad=True)
    z = 4 / y
    x = (z._prev - {y}).pop()
    x.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    
    assert z.data == 2.0
    assert z._op == Operation.DIVISION
    assert z._prev == {x, y}
    assert isclose(x._grad, 1.0 / y.data)
    assert isclose(y._grad, -x.data / (y.data**2))


def test__rtruediv__float() -> None:
    """Test the __rtruediv__ method with an argument of type float."""
    y = Scalar(2.0, requires_grad=True)
    z = 4.0 / y
    x = (z._prev - {y}).pop()
    x.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    
    assert z.data == 2.0
    assert z._op == Operation.DIVISION
    assert z._prev == {x, y}
    assert isclose(x._grad, 1.0 / y.data)
    assert isclose(y._grad, -x.data / (y.data**2))


def test__rtruediv__not_supported() -> None:
    """Test the __rtruediv__ method with an argument with a type that is not supported."""
    y = Scalar(2.0, requires_grad=True)
    with raises(TypeError):
        "4.0" / y

"""Test suit for the method __truediv__ of the Scalar object."""

from math import isclose
from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet
from pytest import raises

def test__truediv__int() -> None:
    """Test the __truediv__ method with an argument of type int."""
    x = Scalar(4.0, requires_grad=True)
    z = x / 2
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    
    assert z.data == 2.0
    assert z._op == Operation.DIVISION
    assert z._prev == OrderedSet([x, y])
    assert isclose(x._grad, 1 / y.data)
    assert isclose(y._grad, -x.data / (y.data**2))


def test__truediv__float() -> None:
    """Test the __truediv__ method with an argument of type float."""
    x = Scalar(4.0, requires_grad=True)
    z = x / 2.0
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    
    assert z.data == 2.0
    assert z._op == Operation.DIVISION
    assert z._prev == OrderedSet([x, y])
    assert isclose(x._grad, 1 / y.data)
    assert isclose(y._grad, -x.data / (y.data**2))


def test__truediv__scalar() -> None:
    """Test the __truediv__ method with an argument of type int."""
    x = Scalar(4.0, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x / y
    z._grad = 1.0
    z._backward_fn()
    
    assert z.data == 2.0
    assert z._op == Operation.DIVISION
    assert z._prev == OrderedSet([x, y])
    assert isclose(x._grad, 1 / y.data)
    assert isclose(y._grad, -x.data / (y.data**2))
    
    
def test__truediv__not_supported() -> None:
    """Test the __truediv__ method with an argument with a type not supported."""
    x = Scalar(4.0, requires_grad=True)
    with raises(TypeError):
        x / "2.0"

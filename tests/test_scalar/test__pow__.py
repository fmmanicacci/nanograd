"""Test suite for the __pow__ method of the Scalar object."""

from math import log, isclose
from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet
from pytest import raises

def test__pow__int() -> None:
    """Test the __pow__ method with an argument of type int."""
    x = Scalar(3.0, requires_grad=True)
    z = x ** 2
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    
    assert isclose(z.data, 9.0)
    assert z._prev == OrderedSet([x, y])
    assert z._op == Operation.EXPONENTIATION
    assert isclose(x._grad, 6.0)
    assert isclose(y._grad, log(x.data) * (x.data ** y.data))


def test__pow__float() -> None:
    """Test the __pow__ method with an argument of type float."""
    x = Scalar(3.0, requires_grad=True)
    z = x ** 2.0
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    
    assert isclose(z.data, 9.0)
    assert z._prev == OrderedSet([x, y])
    assert z._op == Operation.EXPONENTIATION
    assert isclose(x._grad, 6.0)
    assert isclose(y._grad, log(x.data) * (x.data ** y.data))


def test__pow__scalar() -> None:
    """Test the __pow__ method with an argument of type Scalar."""
    x = Scalar(3.0, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x ** y
    z._grad = 1.0
    z._backward_fn()
    
    assert isclose(z.data, 9.0)
    assert z._prev == OrderedSet([x, y])
    assert z._op == Operation.EXPONENTIATION
    assert isclose(x._grad, 6.0)
    assert isclose(y._grad, log(x.data) * (x.data ** y.data))


def test__pow__unsupported() -> None:
    """Test the __pow__ method with an argument of type that is not supported."""
    x = Scalar(3.0, requires_grad=True)
    
    with raises(TypeError):
        x ** "2.0"

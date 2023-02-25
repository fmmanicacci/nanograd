"""Test suit for the __mul__ method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__mul__int() -> None:
    """Test the __mul__ method with an int as argument."""
    x = Scalar(2.0, requires_grad=True)
    z = x * 2
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == y.data
    assert y._grad == x.data


def test__mul__float() -> None:
    """Test the __mul__ method with a float as argument."""
    x = Scalar(2.0, requires_grad=True)
    z = x * 2.0
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == y.data
    assert y._grad == x.data


def test__mul__scalar() -> None:
    """Test the __mul__ method with a Scalar as argument."""
    x = Scalar(2.0, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x * y
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == y.data
    assert y._grad == x.data


def test__mul__not_supported() -> None:
    """Test the __mul__ method with a not supported argument."""
    x = Scalar(2.0, requires_grad=True)
    with raises(TypeError):
        x * "2"

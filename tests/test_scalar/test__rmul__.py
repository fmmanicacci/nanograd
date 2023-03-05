"""Test suit for the __rmul__ method of the Scalar object."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet
from pytest import raises

def test__rmul__int() -> None:
    """Test the __rmul__ method with an int as argument."""
    x = Scalar(2.0, requires_grad=True)
    z = 2 * x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert z._prev == OrderedSet([y, x])
    assert x._grad == y.data
    assert y._grad == x.data


def test__rmul__float() -> None:
    """Test the __rmul__ method with a float as argument."""
    x = Scalar(2.0, requires_grad=True)
    z = 2.0 * x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()
    assert z.data == 4.0
    assert z._op == Operation.MULTIPLICATION
    assert z._prev == OrderedSet([y, x])
    assert x._grad == y.data
    assert y._grad == x.data


def test__rmul__not_supported() -> None:
    """Test the __rmul__ method with a not supported argument."""
    x = Scalar(2.0, requires_grad=True)
    with raises(TypeError):
        "2" * x

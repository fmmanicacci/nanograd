"""Test suit for the __sub__ method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__sub__int() -> None:
    """Test the __sub__ method with an int."""
    x = Scalar(1, requires_grad=True)
    z = x - 2
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == -1
    assert z._op == Operation.SUBTRACTION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == 1.0
    assert y._grad == -1.0


def test__sub__float() -> None:
    """Test the __sub__ method with a float."""
    x = Scalar(1.0, requires_grad=True)
    z = x - 2.0
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == -1.0
    assert z._op == Operation.SUBTRACTION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == 1.0
    assert y._grad == -1.0


def test__sub__Scalar() -> None:
    """Test the __sub__ method with a Scalar."""
    x = Scalar(1.0, requires_grad=True)
    y = Scalar(2.0, requires_grad=True)
    z = x - y
    z._grad = 1.0
    z._backward_fn()

    assert z.data == -1.0
    assert z._op == Operation.SUBTRACTION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == 1.0
    assert y._grad == -1.0


def test__sub__unsupported() -> None:
    """Test the __sub__ method with an unsupported type."""
    x = Scalar(1.0, requires_grad=True)

    with raises(TypeError):
        x - "2"
    
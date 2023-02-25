"""Test suit for the __rsub__ method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__rsub__int() -> None:
    """Test the __rsub__ method with an int."""
    x = Scalar(2, requires_grad=True)
    z = 1 - x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == -1
    assert z._op == Operation.SUBTRACTION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == -1.0
    assert y._grad == 1.0

def test__rsub__float() -> None:
    """Test the __rsub__ method with a float."""
    x = Scalar(2.0, requires_grad=True)
    z = 1.0 - x
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert z.data == -1.0
    assert z._op == Operation.SUBTRACTION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == -1.0
    assert y._grad == 1.0


def test__rsub__scalar() -> None:
    """Test the __rsub__ method with a Scalar."""
    x = Scalar(2.0, requires_grad=True)
    y = Scalar(1.0, requires_grad=True)
    z = y - x
    z._grad = 1.0
    z._backward_fn()

    assert z.data == -1.0
    assert z._op == Operation.SUBTRACTION
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev
    assert x._grad == -1.0
    assert y._grad == 1.0


def test__rsub__unsupported() -> None:
    """Test the __rsub__ method with an unsupported type."""
    x = Scalar(2.0, requires_grad=True)
    with raises(TypeError):
        z = "1.0" - x
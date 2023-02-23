"""Test suit for the __add__ method of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__add__int() -> None:
    """Test that the addition operator works when the argument is an int."""
    x = Scalar(1.0, requires_grad=True)
    y = x + 1
    other = (y._prev - {x}).pop()
    # Check that the output is correct.
    assert y.data == 2.0
    assert y.requires_grad
    assert y._op == Operation.ADDITION
    # Check that the children of the output are correct.
    assert len(y._prev) == 2
    assert x in y._prev
    assert other.data == 1
    assert not other.requires_grad
    assert other._op == Operation.NONE
    assert len(other._prev) == 0
    assert other.label == None


def test__add__float() -> None:
    """Test that the addition operator works when the argument is a float."""
    x = Scalar(1.0, requires_grad=True)
    y = x + 1.0
    other = (y._prev - {x}).pop()
    # Check that the output is correct.
    assert y.data == 2.0
    assert y.requires_grad
    assert y._op == Operation.ADDITION
    # Check that the children of the output are correct.
    assert len(y._prev) == 2
    assert x in y._prev
    assert other.data == 1.0
    assert not other.requires_grad
    assert other._op == Operation.NONE
    assert len(other._prev) == 0
    assert other.label == None


def test__add__scalar() -> None:
    """Test that the addition operator works when the argument is a Scalar."""
    x = Scalar(1.0, requires_grad=True)
    y = Scalar(1.0, requires_grad=True)
    z = x + y
    # Check that the output is correct.
    assert z.data == 2.0
    assert z.requires_grad
    assert z._op == Operation.ADDITION
    # Check that the children of the output are correct.
    assert len(z._prev) == 2
    assert x in z._prev
    assert y in z._prev


def test__add__unsupported_type() -> None:
    """Test that the addition operator raises a TypeError when the argument is not supported."""
    x = Scalar(1.0, requires_grad=True)
    with raises(TypeError):
        x + "1"


def test__add__backward_fn() -> None:
    """Test that the backward function of the addition operator is correct."""
    x = Scalar(1.0, requires_grad=True)
    y = Scalar(1.0, requires_grad=True)
    z = x + y
    z._grad = 1.0
    z._backward_fn()
    assert x._grad == 1.0
    assert y._grad == 1.0
    
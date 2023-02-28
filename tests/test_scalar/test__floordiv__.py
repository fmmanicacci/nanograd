"""Test suite for the method __floordiv__ of the Scalar object."""

from pytest import raises
from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test__floordiv__int_int() -> None:
    """Test the floor division operator with an integer."""
    x = Scalar(21, requires_grad=True)
    z = x // 7
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert type(z.data) == int
    assert z.data == 3
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {x, y}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test__floordiv__int_float() -> None:
    """Test the floor division operator with a float."""
    x = Scalar(21, requires_grad=True)
    z = x // 6.239083
    y = (z._prev - {x}).pop()
    y.requires_grad_(True)
    z._grad = 1.0
    z._backward_fn()

    assert type(z.data) == float
    assert z.data == 3.0
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {x, y}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test__floordiv__int_scalar() -> None:
    """Test the floor division operator with a Scalar."""
    x = Scalar(21, requires_grad=True)
    y = Scalar(6.239083, requires_grad=True)
    z = x // y
    z._grad = 1.0
    z._backward_fn()

    assert type(z.data) == float
    assert z.data == 3.0
    assert z._op == Operation.FLOOR_DIVISION
    assert z._prev == {x, y}
    assert x._grad == 0.0
    assert y._grad == 0.0


def test__floordiv__unsupported_type() -> None:
    """Test the floor division operator with an unsupported type."""
    x = Scalar(21, requires_grad=True)
    with raises(TypeError):
        x // "2.0"

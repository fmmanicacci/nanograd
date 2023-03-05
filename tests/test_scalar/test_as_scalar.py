"""Test suit for the static method `as_scalar`."""

from nanograd.scalar import Scalar
from pytest import raises

def test_as_scalar_int() -> None:
    a = 21
    assert isinstance(a, int)
    try:
        b = Scalar.as_scalar(a)
        assert isinstance(b, Scalar)
    except TypeError:
        assert False


def test_as_scalar_float() -> None:
    a = 21.0
    assert isinstance(a, float)
    try:
        b = Scalar.as_scalar(a)
        assert isinstance(b, Scalar)
    except TypeError:
        assert False


def test_as_scalar_scalar() -> None:
    a = Scalar(21)
    assert isinstance(a, Scalar)
    try:
        b = Scalar.as_scalar(a)
        assert isinstance(b, Scalar)
    except TypeError:
        assert False


def test_as_scalar_unsupported() -> None:
    a = "21"
    assert not isinstance(a, (int, float, Scalar))
    with raises(TypeError):
        Scalar.as_scalar(a)

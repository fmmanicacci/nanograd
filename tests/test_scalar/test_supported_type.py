"""Test suit for the supported types static method."""

from pytest import raises
from nanograd.scalar import Scalar

def test_check_type_int() -> None:
    a = 21
    assert isinstance(a, int)
    try:
        Scalar.supported_type(a)
    except TypeError:
        assert False
        

def test_check_type_float() -> None:
    a = 21.0
    assert isinstance(a, float)
    try:
        Scalar.supported_type(a)
    except TypeError:
        assert False


def test_check_type_scalar() -> None:
    a = Scalar(21)
    assert isinstance(a, Scalar)
    try:
        Scalar.supported_type(a)
    except TypeError:
        assert False


def test_check_type_unsupported() -> None:
    a = "21"
    assert not isinstance(a, (int, float, Scalar))
    with raises(TypeError):
        Scalar.supported_type(a)

"""Test suit for the Scalar object."""

from pytest import raises

from nanograd.scalar import Scalar
from nanograd.enums import Operation


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


def test_str_no_label_op_none() -> None:
    a = Scalar(21)
    str_repr = a.__str__()
    exp_repr = "Scalar(data=21.000000)"
    assert str_repr == exp_repr


def test_str_with_label_op_none() -> None:
    a = Scalar(21, label="a")
    str_repr = a.__str__()
    exp_repr = "Scalar(label=a, data=21.000000)"
    assert str_repr == exp_repr


def test_str_no_label_op_identity() -> None:
    a = Scalar(21, _op=Operation.IDENTITY, _prev={Scalar(21),})
    str_repr = a.__str__()
    exp_repr = "Scalar(data=21.000000, op=identity, prev=1)"
    assert str_repr == exp_repr

  
def test_str_with_label_op_identity() -> None:
    a = Scalar(21, label="a", _op=Operation.IDENTITY, _prev={Scalar(21),})
    str_repr = a.__str__()
    exp_repr = "Scalar(label=a, data=21.000000, op=identity, prev=1)"
    assert str_repr == exp_repr


def test_str_with_label_op_identity_requires_grad() -> None:
    a = Scalar(21, label="a", requires_grad=True, _op=Operation.IDENTITY, _prev={Scalar(21),})
    str_repr = a.__str__()
    exp_repr = "Scalar(label=a, data=21.000000, requires_grad=True, grad=0.000000, op=identity, prev=1)"
    assert str_repr == exp_repr


def test_repr_no_label_op_none() -> None:
    a = Scalar(21)
    repr_str = a.__repr__()
    exp_repr = (
        "Scalar\n"
        "   label         : None\n"
        "   data          : 21.000000\n"
        "   requires_grad : False\n"
        "   grad          : 0.000000\n"
        "   op            : none\n"
        "   prev          : None"
    )
    assert repr_str == exp_repr


def test_repr_with_label_op_none() -> None:
    a = Scalar(21, label="a")
    repr_str = a.__repr__()
    exp_repr = (
        "Scalar\n"
        "   label         : a\n"
        "   data          : 21.000000\n"
        "   requires_grad : False\n"
        "   grad          : 0.000000\n"
        "   op            : none\n"
        "   prev          : None"
    )
    assert repr_str == exp_repr


def test_repr_no_label_op_identity() -> None:
    a = Scalar(21, _op=Operation.IDENTITY, _prev={Scalar(21),})
    repr_str = a.__repr__()
    exp_repr = (
        "Scalar\n"
        "   label         : None\n"
        "   data          : 21.000000\n"
        "   requires_grad : False\n"
        "   grad          : 0.000000\n"
        "   op            : identity\n"
        "   prev          : \n"
        "      Scalar(data=21.000000)"
    )
    assert repr_str == exp_repr


def test_repr_with_label_op_identity_requires_grad() -> None:
    a = Scalar(21, label="a", requires_grad=True, _op=Operation.IDENTITY, _prev={Scalar(21),})
    repr_str = a.__repr__()
    exp_repr = (
        "Scalar\n"
        "   label         : a\n"
        "   data          : 21.000000\n"
        "   requires_grad : True\n"
        "   grad          : 0.000000\n"
        "   op            : identity\n"
        "   prev          : \n"
        "      Scalar(data=21.000000)"
    )
    assert repr_str == exp_repr

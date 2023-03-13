"""Test suit for the __repr__ method."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation
from ordered_set import OrderedSet

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
    a = Scalar(21, _op=Operation.IDENTITY, _prev=OrderedSet([Scalar(21)]))
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
    a = Scalar(
        21,
        label="a",
        requires_grad=True,
        _op=Operation.IDENTITY,
        _prev=OrderedSet([Scalar(21)])
    )
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

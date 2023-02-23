"""Test suit for the Scalar object."""

from nanograd.scalar import Scalar
from nanograd.enums import Operator

def test_str_no_label_op_none() -> None:
    str_repr = Scalar(42).__str__()
    exp_repr = "Value(data=42.000000)"
    assert str_repr == exp_repr


def test_str_with_label_op_none() -> None:
    str_repr = Scalar(42, label="a").__str__()
    exp_repr = "Value(label=a, data=42.000000)"
    assert str_repr == exp_repr


def test_str_no_label_op_add() -> None:
    str_repr = Scalar(42, _op=Operator.ADD).__str__()
    exp_repr = "Value(data=42.000000, grad=0.000000, op=add, prev=0)"
    assert str_repr == exp_repr

  
def test_str_with_label_op_add() -> None:
    str_repr = Scalar(42, label="a", _op=Operator.ADD).__str__()
    exp_repr = "Value(label=a, data=42.000000, grad=0.000000, op=add, prev=0)"
    assert str_repr == exp_repr

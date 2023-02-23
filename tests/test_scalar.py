"""Test suit for the Scalar object."""

from nanograd.scalar import Scalar
from nanograd.enums import Operation

def test_str_no_label_op_none() -> None:
    a = Scalar(21)
    str_repr = a.__str__()
    exp_repr = "Value(data=21.000000)"
    assert str_repr == exp_repr


def test_str_with_label_op_none() -> None:
    a = Scalar(21, label="a")
    str_repr = a.__str__()
    exp_repr = "Value(label=a, data=21.000000)"
    assert str_repr == exp_repr


def test_str_no_label_op_identity() -> None:
    a = Scalar(21, _op=Operation.IDENTITY, _prev={Scalar(21),})
    str_repr = a.__str__()
    exp_repr = "Value(data=21.000000, op=identity, prev=1)"
    assert str_repr == exp_repr

  
def test_str_with_label_op_identity() -> None:
    a = Scalar(21, label="a", _op=Operation.IDENTITY, _prev={Scalar(21),})
    str_repr = a.__str__()
    exp_repr = "Value(label=a, data=21.000000, op=identity, prev=1)"
    assert str_repr == exp_repr


def test_str_with_label_op_identity_requires_grad() -> None:
    a = Scalar(21, label="a", requires_grad=True, _op=Operation.IDENTITY, _prev={Scalar(21),})
    str_repr = a.__str__()
    exp_repr = "Value(label=a, data=21.000000, grad=0.000000, op=identity, prev=1)"
    assert str_repr == exp_repr

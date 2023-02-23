"""Definition of the Scalar object."""

from .enums import Operation

class Scalar:
    """
    A Scalar object represents a node in the computational graph. It is central piece
    of the autograd engine since it is in charge of computing the gradient during the
    backward phase.
    """
    
    def __init__(
        self,
        data: int | float,
        label: str | None = None,
        requires_grad: bool = False,
        _prev: set['Scalar'] = None,
        _op: Operation = Operation.NONE
    ) -> None:
        """Constructor."""
        self.data = data
        self.label = label
        self.requires_grad = requires_grad
        self._grad = 0.0
        self._prev = set() if _prev is None else _prev
        self._op = _op
        self.backward_fn = lambda: None
        
    def __str__(self) -> str:
        """Provide a string representation of the object."""
        label_str = "" if self.label is None else f"label={self.label}, "
        data_str = f"data={self.data:.6f}"
        grad_str = "" if not self.requires_grad else f", grad={self._grad:.6f}"
        op_str = "" if self._op is Operation.NONE else f", op={self._op.value}"
        prev_str = "" if self._op is Operation.NONE else f", prev={len(self._prev)}"
        return f"Value({label_str}{data_str}{grad_str}{op_str}{prev_str})"
        
    def __repr__(self) -> str:
        """Provide an information-rich string representation of the object."""
        ...


if __name__ == '__main__':
    print(Scalar(42, _op=Operation.ADD))
    
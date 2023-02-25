"""Definition of the Scalar object."""

from typing import Union
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
        self._backward_fn = lambda: None
        self._backward = 0.0
        self.requires_grad_(requires_grad)

    @staticmethod
    def supported_type(x: Union[int, float, 'Scalar']) -> None:
        """Check that the type of the argument `x` is supported."""
        if not isinstance(x, (Scalar, int, float)):
            raise TypeError(f"The following type {type(x)} is not supported.")

    @staticmethod
    def as_scalar(
        x: int | float,
        label: str | None = None,
        requires_grad: bool = False
    ) -> 'Scalar':
        """Return the argument `x` as a Scalar if it is possible."""
        Scalar.supported_type(x)
        if isinstance(x, Scalar):
            return x
        elif isinstance(x, (int, float)):
            return Scalar(x, label=label, requires_grad=requires_grad)

    def requires_grad_(self, requires_grad: bool) -> None:
        """Set the `requires_grad` attribute of the object."""
        self.requires_grad = requires_grad
        # If the object requires gradient, we need to set the `_backward` attribute to 1.0
        # so that the gradient is computed during the backward phase.
        # Otherwise, we set it to 0.0 so that the gradient is not computed.
        self._backward = 1.0 if requires_grad else 0.0

    def add(self, other: Union[int, float, 'Scalar'], label: str | None = None) -> 'Scalar':
        """Addition operator."""
        out = self + other
        out.label = label
        return out
    
    def neg(self, label: str | None = None) -> 'Scalar':
        """Negation operator."""
        out = -self
        out.label = label
        return out
    
    def sub(self, other: Union[int, float, 'Scalar'], label: str | None = None) -> 'Scalar':
        """Subtraction operator."""
        out = self - other
        out.label = label
        return out
    
    def __neg__(self) -> 'Scalar':
        """Negation operator."""
        # Perform the negation.
        out = Scalar(
            -self.data,
            requires_grad=True,
            _prev={self},
            _op=Operation.NEGATION
        )
        # Define the backward function
        def _backward_fn() -> None:
            self._grad += self._backward * (-1.0) * out._grad
        out._backward_fn = _backward_fn
        return out

    def __sub__(self, other: Union[int, float, 'Scalar']) -> 'Scalar':
        """Subtraction operator."""
        # Check that the type of the argument is supported and cast it to a Scalar if necessary.
        other = Scalar.as_scalar(other)
        # Perform the subtraction.
        out = Scalar(
            self.data - other.data,
            requires_grad=True,
            _prev={self, other},
            _op=Operation.SUBTRACTION
        )
        # Define the backward function: the gradient of each operand is the gradient of the output
        def _backward_fn() -> None:
            self._grad += self._backward * (1.0) * out._grad
            other._grad += other._backward * (-1.0) * out._grad
        out._backward_fn = _backward_fn
        return out
    
    def __rsub__(self, other: int | float) -> 'Scalar':
        """Right subtraction operator."""
        other = Scalar.as_scalar(other)
        out = other - self
        return out
    
    def __add__(self, other: Union[int, float, 'Scalar']) -> 'Scalar':
        """Addition operator."""
        # Check that the type of the argument is supported and cast it to a Scalar if necessary.
        other = Scalar.as_scalar(other)
        # Perform the addition.
        out = Scalar(
            self.data + other.data,
            requires_grad=True,
            _prev={self, other},
            _op=Operation.ADDITION
        )
        # Define the backward function: the gradient of each operand is the gradient of the output
        def _backward_fn() -> None:
            self._grad += self._backward * (1.0) * out._grad
            other._grad += other._backward * (1.0) * out._grad
        out._backward_fn = _backward_fn
        return out

    def __radd__(self, other: Union[int, float, 'Scalar']) -> 'Scalar':
        """Right addition operator."""
        # Addition is commutative, so we can use the __add__ method.
        return self + other

    def __str__(self) -> str:
        """Provide a string representation of the object."""
        label_str = "" if self.label is None else f"label={self.label}, "
        data_str = f"data={self.data:.6f}"
        req_grad_str = "" if not self.requires_grad else f", requires_grad={self.requires_grad}"
        grad_str = "" if not self.requires_grad else f", grad={self._grad:.6f}"
        op_str = "" if self._op is Operation.NONE else f", op={self._op.value}"
        prev_str = "" if self._op is Operation.NONE else f", prev={len(self._prev)}"
        return f"Scalar({label_str}{data_str}{req_grad_str}{grad_str}{op_str}{prev_str})"
        
    def __repr__(self) -> str:
        """Provide an information-rich string representation of the object."""
        start_str = "Scalar\n"
        label_str = f"{' ':3}label         : {self.label}\n"
        data_str = f"{' ':3}data          : {self.data:.6f}\n"
        requires_grad_str = f"{' ':3}requires_grad : {self.requires_grad}\n"
        grad_str = f"{' ':3}grad          : {self._grad:.6f}\n"
        op_str = f"{' ':3}op            : {self._op.value}\n"
        children_str = ''.join([f"\n{' ':6}{child}" for child in self._prev]) if len(self._prev) > 0 else "None"
        prev_str = f"{' ':3}prev          : {children_str}"

        return f"{start_str}{label_str}{data_str}{requires_grad_str}{grad_str}{op_str}{prev_str}"

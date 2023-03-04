"""This module contains all the enumerations defined for nanograd."""

from enum import Enum

class Operation(Enum):
    """
    Enumeration of the operator that can be used to build the computational
    graph with nanograd.
    """
    
    NONE = 'none'
    IDENTITY = 'identity'
    ADDITION = 'add'
    NEGATION = 'neg'
    SUBTRACTION = 'sub'
    MULTIPLICATION = 'mul'
    DIVISION = 'div'
    FLOOR_DIVISION = 'floordiv'
    INVERTION = 'inv'
    EXPONENTIATION = 'pow'
    EXPONENTIAL = 'exp'
    HYPERBOLIC_TANGENT = 'tanh'
    RELU = 'relu'

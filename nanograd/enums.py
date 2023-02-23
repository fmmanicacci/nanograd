"""This module contains all the enumerations defined for nanograd."""

from enum import Enum

class Operator(Enum):
    """
    Enumeration of the operator that can be used to build the computational
    graph with nanograd.
    """
    
    NONE = 'none'
    ADD = 'add'
    
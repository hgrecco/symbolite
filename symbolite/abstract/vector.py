"""
    symbolite.abstract.vector
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Objects and functions for vector operations.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

import dataclasses

from .scalar import Scalar
from .symbol import Function, OperandMixin, Symbol
from ..core.base import Unsupported


@dataclasses.dataclass(frozen=True)
class Vector(Symbol):
    """A user defined symbol."""

    namespace = ""

    def __getattr__(self, key):
        return Unsupported


@dataclasses.dataclass(frozen=True)
class VectorFunction(Function):
    namespace = "vector"


sum = VectorFunction("sum", arity=1)
prod = VectorFunction("prod", arity=1)


def vectorize(
    expr: OperandMixin,
    symbol_names: tuple[str, ...] | dict[str, int],
    varname="arr",
) -> OperandMixin:
    """Vectorize expression by replacing test_scalar symbols
    by an array at a given indices.

    Parameters
    ----------
    expr
    symbol_names
        if a tuple, provides the names of the symbols
        which will be mapped to the indices given by their position.
        if a dict, maps symbol names to indices.
    varname
        name of the array variable
    """
    if isinstance(symbol_names, dict):
        it = zip(symbol_names.values(), symbol_names.keys())
    else:
        it = enumerate(symbol_names)

    arr = Vector(varname)

    reps = {Scalar(name): arr[ndx] for ndx, name in it}
    return expr.subs(reps)


def auto_vectorize(expr, varname="vec") -> tuple[tuple[str, ...], OperandMixin]:
    """Vectorize expression by replacing all test_scalar symbols
    by an array at a given indices. Symbols are ordered into
    the array alphabetically.

    Parameters
    ----------
    expr
    varname
        name of the array variable

    Returns
    -------
    tuple[str, ...]
        symbol names as ordered in the array.
    SymbolicExpression
        vectorized expression.
    """
    symbol_names = tuple(sorted(expr.symbol_names("")))
    return symbol_names, vectorize(expr, symbol_names, varname)

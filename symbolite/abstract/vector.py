"""
    symbolite.abstract.vector
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Objects and functions for vector operations.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Sequence, overload

from .scalar import NumberT, Scalar
from .symbol import BaseFunction, Symbol

VectorT = Iterable[NumberT]


@dataclasses.dataclass(frozen=True)
class Vector(Symbol):
    """A user defined symbol."""

    def __getitem__(self, key: int | Scalar) -> Scalar:
        return super().__getitem__(key)

    def __getattr__(self, key: Any):
        raise AttributeError(key)


@dataclasses.dataclass(frozen=True)
class CumulativeFunction(BaseFunction):
    namespace: str = "vector"
    arity: int = 1

    def __call__(self, arg1: Vector | VectorT) -> Scalar:
        return super()._call(arg1)  # type: ignore


sum = CumulativeFunction("sum", namespace="vector")
prod = CumulativeFunction("prod", namespace="vector")


@overload
def vectorize(
    expr: NumberT,
    symbol_names: Sequence[str] | Mapping[str, int],
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> NumberT:
    ...


@overload
def vectorize(
    expr: Symbol,
    symbol_names: Sequence[str] | Mapping[str, int],
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> Symbol:
    ...


@overload
def vectorize(
    expr: Iterable[NumberT | Symbol],
    symbol_names: Sequence[str] | Mapping[str, int],
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> tuple[NumberT | Symbol, ...]:
    ...


def vectorize(
    expr: NumberT | Symbol | Iterable[NumberT | Symbol],
    symbol_names: Sequence[str] | Mapping[str, int],
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> NumberT | Symbol | tuple[NumberT | Symbol, ...]:
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
    if isinstance(expr, NumberT):
        return expr

    if not isinstance(expr, Symbol):
        return tuple(vectorize(symbol, symbol_names, varname) for symbol in expr)

    if isinstance(symbol_names, dict):
        it = zip(symbol_names.values(), symbol_names.keys())
    else:
        it = enumerate(symbol_names)

    arr = Vector(varname)

    reps = {scalar_type(name): arr[ndx] for ndx, name in it}
    return expr.subs(reps)


@overload
def auto_vectorize(
    expr: NumberT,
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> tuple[tuple[str, ...], Symbol]:
    ...


@overload
def auto_vectorize(
    expr: Symbol,
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> tuple[tuple[str, ...], Symbol]:
    ...


@overload
def auto_vectorize(
    expr: Iterable[Symbol],
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> tuple[tuple[str, ...], tuple[Symbol, ...]]:
    ...


def auto_vectorize(
    expr: NumberT | Symbol | Iterable[Symbol],
    varname: str = "vec",
    scalar_type: type[Scalar] = Scalar,
) -> tuple[tuple[str, ...], NumberT | Symbol | tuple[NumberT | Symbol, ...]]:
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
    if isinstance(expr, NumberT):
        return tuple(), expr

    if not isinstance(expr, Symbol):
        expr = tuple(expr)
        out = set[str]()
        for symbol in expr:
            out.update(symbol.symbol_names(""))
        symbol_names = tuple(sorted(out))
        return symbol_names, vectorize(expr, symbol_names, varname, scalar_type)
    else:
        symbol_names = tuple(sorted(expr.symbol_names("")))
        return symbol_names, vectorize(expr, symbol_names, varname, scalar_type)

"""
symbolite.core.symbolgroup
~~~~~~~~~~~~~~~~~~~~~~~~~~

Groups of symbols and symbolic expressions.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Iterable, Mapping

from ..abstract.symbol import Symbol, yield_named
from .operations import evaluate, evaluate_impl, substitute, substitute_by_name


class SymbolicList(list[Symbol]):
    @classmethod
    def from_iterable(cls, it: Iterable[Symbol]):
        return cls(it)

    def __str__(self):
        return "\n".join(str(se) for se in self)


@substitute.register
def _(self: SymbolicList, *mappers: Mapping[Any, Any]) -> SymbolicList:
    """Replace symbols, functions, values, etc by others.

    If multiple mappers are provided,
        they will be used in order (using a ChainMap)

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    *mappers
        dictionaries mapping source to destination objects.
    """
    return self.__class__.from_iterable((substitute(se, *mappers) for se in self))


@substitute_by_name.register
def _(self: SymbolicList, **symbols: Any) -> SymbolicList:
    """Replace Symbols by values or objects, matching by name.

    If multiple mappers are provided,
        they will be used in order (using a ChainMap)

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    **symbols
        keyword arguments connecting names to values.
    """
    return self.__class__.from_iterable(
        (substitute_by_name(se, **symbols) for se in self)
    )


@evaluate_impl.register
def _(self: SymbolicList, **libs: types.ModuleType) -> SymbolicList:
    """Evaluate expression.

    If no implementation library is provided:
    1. 'libsl' will be looked up going back though the stack
        until is found.
    2. If still not found, the implementation using the python
        math module will be used (and a warning will be issued).

    Parameters
    ----------
    libs
        implementations
    """

    return self.__class__.from_iterable(evaluate(se, **libs) for se in self)


class SymbolicNamespace:
    expressions: SymbolicList = SymbolicList()

    # TODO: remove this.
    @classmethod
    def symbol_names(cls, namespace: str | None = "") -> set[str]:
        out = set()
        for expr in cls.expressions:
            out.update((named.name for named in yield_named(expr, namespace)))
        return out


@dataclasses.dataclass(frozen=True, repr=False)
class AutoSymbol(Symbol):
    name: str = "<auto>"

    def __set_name__(self, owner: Any, name: str):
        if issubclass(owner, SymbolicNamespace):
            object.__setattr__(self, "name", name)
            owner.expressions.append(self)

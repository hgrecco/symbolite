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
from typing import Any, Generator, Iterable, Mapping

from ..abstract.symbol import Symbol
from ..core.named import Named, yield_named
from .operations import evaluate_impl, substitute, substitute_by_name


class SymbolicList(list[Symbol]):
    @classmethod
    def from_iterable(cls, it: Iterable[Symbol]):
        return cls(it)

    def __str__(self):
        return "\n".join(str(se) for se in self)


@substitute.register
def _(self: SymbolicList, replacements: Mapping[Any, Any]) -> SymbolicList:
    return self.__class__.from_iterable((substitute(se, replacements) for se in self))


@substitute_by_name.register
def _(self: SymbolicList, **replacements: Any) -> SymbolicList:
    return self.__class__.from_iterable(
        (substitute_by_name(se, **replacements) for se in self)
    )


@evaluate_impl.register
def _(self: SymbolicList, libsl: types.ModuleType) -> SymbolicList:
    return self.__class__.from_iterable(evaluate_impl(se, libsl) for se in self)


@yield_named.register
def _(
    self: SymbolicList, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    for se in self:
        yield from yield_named(se, include_anonymous)


# This is necessary to use singledispatch on classes.
class SymbolicNamespaceMeta(type):
    expressions: SymbolicList


class SymbolicNamespace(metaclass=SymbolicNamespaceMeta):
    expressions: SymbolicList = SymbolicList()


@yield_named.register
def _(
    self: SymbolicNamespace, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    for expr in self.expressions:
        yield from yield_named(expr, include_anonymous)


@yield_named.register
def _(
    self: SymbolicNamespaceMeta, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    for expr in self.expressions:
        yield from yield_named(expr, include_anonymous)


@dataclasses.dataclass(frozen=True, repr=False)
class AutoSymbol(Symbol):
    name: str = "<auto>"

    def __set_name__(self, owner: Any, name: str):
        if issubclass(owner, SymbolicNamespace):
            object.__setattr__(self, "name", name)
            owner.expressions.append(self)

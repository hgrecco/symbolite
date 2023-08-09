from __future__ import annotations

import dataclasses
import types
from typing import Any, Iterable, Mapping

from ..abstract.symbol import Symbol


class SymbolicList(list[Symbol]):
    @classmethod
    def from_iterable(cls, it: Iterable[Symbol]):
        return cls(it)

    def subs(self, *mappers: Mapping[Any, Any]) -> SymbolicList:
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
        return self.__class__.from_iterable((se.subs(*mappers) for se in self))

    def subs_by_name(self, **symbols: Any) -> SymbolicList:
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
        return self.__class__.from_iterable((se.subs_by_name(**symbols) for se in self))

    def eval(self, **libs: types.ModuleType) -> SymbolicList:
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

        return self.__class__.from_iterable(se.eval(**libs) for se in self)

    def symbol_names(self, namespace: str | None = "") -> set[str]:
        """Return a set of symbol names (with full namespace indication).

        Parameters
        ----------
        namespace: str or None
            If None, all symbols will be returned independently of the namespace.
            If a string, will compare Symbol.namespace to that.
            Defaults to "" which is the namespace for user defined symbols.
        """
        s: set[str] = set()
        return s.union(*(se.symbol_names(namespace) for se in self))

    def __str__(self):
        return "\n".join(str(se) for se in self)


class SymbolicNamespace:
    expressions: SymbolicList = SymbolicList()

    @classmethod
    def symbol_names(cls, namespace: str | None = "") -> set[str]:
        return cls.expressions.symbol_names(namespace)


@dataclasses.dataclass(frozen=True)
class AutoSymbol(Symbol):
    name: str = "<auto>"

    def __set_name__(self, owner: Any, name: str):
        if issubclass(owner, SymbolicNamespace):
            object.__setattr__(self, "name", name)
            owner.expressions.append(self)

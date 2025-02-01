"""
symbolite.core.symbolgroup
~~~~~~~~~~~~~~~~~~~~~~~~~~

Groups of symbols and symbolic expressions.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import inspect
import types
import warnings
from typing import Any, Generator, Mapping

from ..abstract.symbol import Symbol
from ..core.named import Named, yield_named
from .operations import as_string, evaluate_impl, substitute


# This is necessary to use singledispatch on classes.
class SymbolicNamespaceMeta(type):
    pass


class SymbolicNamespace(metaclass=SymbolicNamespaceMeta):
    pass


# In Python 3.10 UnionTypes are not supported by singledispatch.register
@yield_named.register(SymbolicNamespaceMeta)
@yield_named.register(SymbolicNamespace)
def _(self, include_anonymous: bool = False) -> Generator[Named, None, None]:
    assert isinstance(self, (SymbolicNamespace, SymbolicNamespaceMeta))
    for name in dir(self):
        if name.startswith("__"):
            continue
        attr = getattr(self, name)
        yield from yield_named(attr, include_anonymous)


@substitute.register(SymbolicNamespaceMeta)
@substitute.register(SymbolicNamespace)
def _(self, replacements: Mapping[Any, Any]) -> Any:
    assert isinstance(self, (SymbolicNamespace, SymbolicNamespaceMeta))

    d = {}
    for attr_name in dir(self):
        if attr_name.startswith("__"):
            continue
        attr = getattr(self, attr_name)
        d[attr_name] = substitute(attr, replacements)

    return type(self.__name__, inspect.getmro(self), d)


@evaluate_impl.register(SymbolicNamespaceMeta)
@evaluate_impl.register(SymbolicNamespace)
def _(self, libsl: types.ModuleType) -> Any:
    return {
        attr_name: evaluate_impl(getattr(self, attr_name), libsl)
        for attr_name in dir(self)
        if not attr_name.startswith("__")
    }


@as_string.register(SymbolicNamespaceMeta)
@as_string.register(SymbolicNamespace)
def _(self) -> str:
    yield_free_symbols: list[str] = []
    lines: list[str] = []

    for attr_name in dir(self):
        if attr_name.startswith("__"):
            continue
        attr = getattr(self, attr_name)
        if not isinstance(attr, Symbol):
            continue

        if attr.name is not None and attr_name != attr.name:
            warnings.warn(f"Missmatched names in attribute {attr_name} vs. {attr}")

        if attr.expression is None:
            if attr.name is not None and attr.name not in yield_free_symbols:
                yield_free_symbols.append(attr.name)
        else:
            lines.append(f"{attr_name} = {attr!s}")

    return "\n".join(lines)


# @as_function.register
# def _(
#     expr: Any,
#     function_name: str,
#     params: Sequence[TH],
#     libsl: types.ModuleType | None = None,
# ) -> Callable[..., Any]:

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
from typing import Any, Generator, Mapping

from ..abstract.symbol import Symbol
from .named import Named
from .operations import (
    as_function_def,
    as_string,
    assign,
    build_function_code,
    evaluate_impl,
    free_symbols,
    substitute,
    yield_named,
)


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
    assert isinstance(self, (SymbolicNamespace, SymbolicNamespaceMeta))
    return {
        attr_name: evaluate_impl(getattr(self, attr_name), libsl)
        for attr_name in dir(self)
        if not attr_name.startswith("__")
    }


@as_string.register(SymbolicNamespaceMeta)
@as_string.register(SymbolicNamespace)
def _(self) -> str:
    assert isinstance(self, (SymbolicNamespace, SymbolicNamespaceMeta))

    lines = [f"# {self.__name__}", ""]

    for fs in free_symbols(self):
        lines.append(assign(fs.name, f"{fs.__class__.__name__}()"))

    lines.append("")

    for attr_name in dir(self):
        attr = getattr(self, attr_name)
        if not isinstance(attr, Symbol):
            continue

        if attr.expression is not None:
            lines.append(assign(attr_name, f"{attr!s}"))

    return "\n".join(lines)


@as_function_def.register(SymbolicNamespaceMeta)
@as_function_def.register(SymbolicNamespace)
def _(
    expr,
) -> str:
    assert isinstance(expr, (SymbolicNamespace, SymbolicNamespaceMeta))

    lines: list[str] = []
    for attr_name in dir(expr):
        attr = getattr(expr, attr_name)
        if not isinstance(attr, Symbol):
            continue

        if attr.expression is not None:
            lines.append(assign(attr_name, f"{attr!s}"))

    return build_function_code(
        expr.__name__,
        map(str, free_symbols(expr)),
        lines,
        [
            "__return",
        ],
    )

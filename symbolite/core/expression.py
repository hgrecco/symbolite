"""
symbolite.core.expression
~~~~~~~~~~~~~~~~~~~~~~~~~

An expression is the result of a function that has been called with
certain arguments.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from typing import TYPE_CHECKING, Any, Generator, Mapping

from .named import Named
from .operations import (
    as_function_def,
    assign,
    build_function_code,
    evaluate_impl,
    free_symbols,
    substitute,
    yield_named,
)
from .util import repr_without_defaults

if TYPE_CHECKING:
    pass


@dataclasses.dataclass(frozen=True, repr=False)
class Expression:
    """A Function that has been called with certain arguments."""

    func: Named
    args: tuple[Any, ...]
    kwargs_items: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.kwargs_items, dict):
            object.__setattr__(self, "kwargs_items", tuple(self.kwargs_items.items()))

    @functools.cached_property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs_items)

    def __str__(self) -> str:
        return self.func.format(*self.args, *self.kwargs)

    def __repr__(self) -> str:
        return repr_without_defaults(self)


@substitute.register
def _(self: Expression, mapper: Mapping[Any, Any]) -> Expression:
    func = mapper.get(self.func, self.func)
    args = tuple(substitute(arg, mapper) for arg in self.args)
    kwargs = {k: substitute(arg, mapper) for k, arg in self.kwargs_items}

    return Expression(func, args, tuple(kwargs.items()))


@evaluate_impl.register
def _(self: Expression, libsl: types.ModuleType) -> Any:
    func = evaluate_impl(self.func, libsl)
    args = tuple(evaluate_impl(arg, libsl) for arg in self.args)
    kwargs = {k: evaluate_impl(arg, libsl) for k, arg in self.kwargs_items}

    try:
        return func(*args, **kwargs)
    except Exception as ex:
        try:
            ex.add_note(f"While evaluating {func}(*{args}, **{kwargs}): {ex}")
        except AttributeError:
            pass
        raise ex


@yield_named.register
def _(
    self: Expression, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    if include_anonymous or self.func.name is not None:
        yield self.func

    for arg in self.args:
        yield from yield_named(arg, include_anonymous)

    for _, v in self.kwargs_items:
        yield from yield_named(v, include_anonymous)


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class NamedExpression(Named):
    """An expression with name and namespace."""

    expression: Expression | None = None


@as_function_def.register
def _(expr: NamedExpression) -> str:
    function_name = expr.name or "f"
    return build_function_code(
        function_name,
        map(str, free_symbols(expr)),
        [
            assign("__out", str(expr)),
        ],
        [
            "__out",
        ],
    )

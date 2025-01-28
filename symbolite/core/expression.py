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
from typing import Any, Generator, Mapping

from .named import Named, yield_named
from .operations import evaluate_impl, substitute, substitute_by_name
from .util import repr_without_defaults


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
    """Replace symbols, functions, values, etc by others.

    If multiple mappers are provided,
        they will be used in order (using a ChainMap)

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    mappers
        dictionary mapping source to destination objects.
    """
    func = mapper.get(self.func, self.func)
    args = tuple(substitute(arg, mapper) for arg in self.args)
    kwargs = {k: substitute(arg, mapper) for k, arg in self.kwargs_items}

    return Expression(func, args, tuple(kwargs.items()))


@substitute_by_name.register
def _(self: Expression, **mapper: Any) -> Expression:
    """Replace symbols, functions, values, etc by others.

    If multiple mappers are provided,
        they will be used in order (using a ChainMap)

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    mappers
        dictionary mapping source to destination objects.
    """
    func = mapper.get(str(self.func), self.func)
    args = tuple(substitute_by_name(arg, **mapper) for arg in self.args)
    kwargs = {k: substitute_by_name(arg, **mapper) for k, arg in self.kwargs_items}

    return Expression(func, args, tuple(kwargs.items()))


@evaluate_impl.register
def _(self: Expression, libsl: types.ModuleType) -> Any:
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
    if include_anonymous or not self.func.is_anonymous:
        yield self.func

    for arg in self.args:
        yield from yield_named(arg, include_anonymous)

    for _, v in self.kwargs_items:
        yield from yield_named(v, include_anonymous)


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class NamedExpression(Named):
    """An expression with name and namespace."""

    expression: Expression | None = None

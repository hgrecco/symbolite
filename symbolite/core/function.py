"""
symbolite.core.function
~~~~~~~~~~~~~~~~~~~~~~~

Symbolic functions.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import types
from operator import attrgetter
from typing import (
    Any,
    Generator,
)

from .expression import Expression, NamedExpression
from .named import Named, yield_named
from .operations import evaluate_impl
from .util import Unsupported


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class BaseFunction(Named):
    """A callable primitive that will return a call."""

    fmt: str | None = None
    arity: int | None = None

    @property
    def output_type(self) -> type[NamedExpression]:
        return NamedExpression

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        return self.output_type(expression=self._build_resolver(*args, **kwargs))

    def _build_resolver(self, *args: Any, **kwargs: Any) -> Expression:
        if self.arity is None:
            return Expression(self, args, tuple(kwargs.items()))
        if kwargs:
            raise ValueError(
                "If arity is given, keyword arguments should not be provided."
            )
        if len(args) != self.arity:
            raise ValueError(
                f"Invalid number of arguments ({len(args)}), expected {self.arity}."
            )
        return Expression(self, args)

    def format(self, *args: Any, **kwargs: Any) -> str:
        if self.fmt:
            return self.fmt.format(*args, **kwargs)

        plain_args = args + tuple(f"{k}={v}" for k, v in kwargs.items())
        return f"{str(self)}({', '.join((str(v) for v in plain_args))})"


@yield_named.register
def _(
    self: BaseFunction, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    yield self


@evaluate_impl.register
def _(expr: BaseFunction, libsl: types.ModuleType) -> Any | Unsupported:
    return attrgetter(str(expr))(libsl)

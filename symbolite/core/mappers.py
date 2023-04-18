"""
    symbolite.core.mappers
    ~~~~~~~~~~~~~~~~~~~~~~

    Visitors iterate through an expression.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import collections
import dataclasses
import types
import typing as ty
from operator import attrgetter

from .base import BaseCall, BaseFunction, BaseSymbol, Unsupported


class Visitor:
    """Base class for all visitors that dispacth to calls, functions,
    symbol, other.
    """

    def __call__(self, value: ty.Any) -> ty.Any:
        if isinstance(value, BaseCall):
            return self.visit_call(value)
        elif isinstance(value, BaseFunction):
            return self.visit_function(value)
        elif isinstance(value, BaseSymbol):
            return self.visit_symbol(value)
        else:
            return self.visit_other(value)

    def visit_call(self, expr: BaseCall) -> ty.Any:
        args = tuple(self(arg) for arg in expr.args)
        kwargs = {k: self(arg) for k, arg in expr.kwargs_items}

        f = self(expr.func)

        return f(*args, **kwargs)

    def visit_function(self, expr: BaseFunction) -> ty.Any:
        return expr

    def visit_symbol(self, expr: BaseSymbol) -> ty.Any:
        return expr

    def visit_other(self, expr: ty.Any) -> ty.Any:
        return expr


class StringifyVisitor(Visitor):
    """String representation of an expression."""

    def visit_function(self, expr: BaseFunction) -> ty.Any:
        return expr.format

    def visit_symbol(self, expr: BaseSymbol) -> ty.Any:
        return str(expr)

    def visit_other(self, expr: ty.Any) -> ty.Any:
        return str(expr)


@dataclasses.dataclass
class CounterVisitor(Visitor):
    """Count the number of appeareance of calls, functions, symbols and other
    in an expression.

    By default, the calls are not counted.

    Access the result through the counter attribute.
    """

    call: bool = False
    function: bool = True
    symbol: bool = True
    other: bool = True

    counter: collections.Counter[ty.Any] = dataclasses.field(
        default_factory=collections.Counter[ty.Any]
    )

    def visit_call(self, expr: BaseCall) -> ty.Any:
        if self.call:
            self.counter[expr] += 1
        return super().visit_call(expr)

    def visit_function(self, expr: BaseFunction) -> ty.Any:
        if self.function:
            self.counter[expr] += 1
        return super().visit_function(expr)

    def visit_symbol(self, expr: BaseSymbol) -> ty.Any:
        if self.symbol:
            self.counter[expr] += 1
        return super().visit_symbol(expr)

    def visit_other(self, expr: ty.Any) -> ty.Any:
        if self.other:
            self.counter[expr] += 1
        return super().visit_other(expr)


@dataclasses.dataclass
class SubstituteVisitor(Visitor):
    """Substitute elements in an expression by other elements

    If `must_exist` is True (not the default), each term MUST
    be in the replacements dictionary.
    """

    replacements: dict[ty.Any, ty.Any]
    must_exist: bool = False

    def visit_function(self, expr: BaseFunction) -> ty.Any:
        if self.must_exist:
            return self.replacements[expr]
        return self.replacements.get(expr, expr)

    def visit_symbol(self, expr: BaseSymbol) -> ty.Any:
        if self.must_exist:
            return self.replacements[expr]
        return self.replacements.get(expr, expr)

    def visit_other(self, expr: ty.Any) -> ty.Any:
        if self.must_exist:
            return self.replacements[expr]
        return self.replacements.get(expr, expr)


@dataclasses.dataclass
class SubstituteByNameVisitor(Visitor):
    """Substitute symbols and functions in an expression by other elements,
    matching by name

    If `must_exist` is True (not the default), each term MUST
    be in the replacements dictionary.
    """

    replacements: dict[str, ty.Any]
    must_exist: bool = False

    def visit_function(self, expr: BaseFunction) -> ty.Any:
        if self.must_exist:
            return self.replacements[expr.name]
        return self.replacements.get(expr.name, expr)

    def visit_symbol(self, expr: BaseSymbol) -> ty.Any:
        if self.must_exist:
            return self.replacements[expr.name]
        return self.replacements.get(expr.name, expr)


@dataclasses.dataclass
class EvaluateVisitor(Visitor):
    """Evaluate an expression for a given implementation library."""

    libsl: types.ModuleType

    def visit_function(self, expr: BaseFunction) -> ty.Any:
        f = attrgetter(str(expr))(self.libsl)

        if f is Unsupported:
            raise Unsupported(f"{expr} is not supported in module {self.libsl}")

        return f

    def visit_symbol(self, expr: BaseSymbol) -> ty.Any:
        if expr.namespace == "":
            # User defined symbol, try to map the class
            name = (
                f"{expr.__class__.__module__.split('.')[-1]}.{expr.__class__.__name__}"
            )
            f = attrgetter(name)(self.libsl)

            if f is Unsupported:
                raise Unsupported(
                    f"{name} is not supported in module {self.libsl.__name__}"
                )

            return f(expr.name)

        else:
            name = str(expr)

            value = attrgetter(name)(self.libsl)

            if value is Unsupported:
                raise Unsupported(
                    f"{name} is not supported in module {self.libsl.__name__}"
                )

            return value

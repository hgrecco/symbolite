"""
    symbolite.operands
    ~~~~~~~~~~~~~~~~~~

    Expression operands.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import types
import warnings

NAMESPACE = "libsl"
LIBPREFIX = NAMESPACE + "."


@functools.lru_cache()
def _lib():
    # Is there a better way to deal with circular imports.
    from . import lib

    return lib


@functools.lru_cache()
def _translators():
    # Is there a better way to deal with circular imports.
    from . import translators

    return translators


class OperandMixin:
    """Base class for objects that might operate with others."""

    def __pos__(self):
        return Call(_lib().op_pos, (self,))

    def __neg__(self):
        return Call(_lib().op_neg, (self,))

    def __abs__(self):
        return Call(_lib().abs, (self,))

    def __add__(self, other):
        return Call(_lib().op_add, (self, other))

    def __sub__(self, other):
        return Call(_lib().op_sub, (self, other))

    def __mul__(self, other):
        return Call(_lib().op_mul, (self, other))

    def __truediv__(self, other):
        return Call(_lib().op_truediv, (self, other))

    def __floordiv__(self, other):
        return Call(_lib().op_floordiv, (self, other))

    def __mod__(self, other):
        return Call(_lib().op_mod, (self, other))

    def __pow__(self, power, modulo=None):
        if modulo is None:
            return Call(_lib().op_pow, (self, power))
        return Call(_lib().op_modpow, (self, power, modulo))

    def __radd__(self, other):
        return Call(_lib().op_add, (other, self))

    def __rsub__(self, other):
        return Call(_lib().op_sub, (other, self))

    def __rmul__(self, other):
        return Call(_lib().op_mul, (other, self))

    def __rtruediv__(self, other):
        return Call(_lib().op_truediv, (other, self))

    def __rfloordiv__(self, other):
        return Call(_lib().op_floordiv, (other, self))

    def __rmod__(self, other):
        return Call(_lib().op_mod, (other, self))

    def __rpow__(self, other):
        return Call(_lib().op_pow, (other, self))


@dataclasses.dataclass(frozen=True)
class SymbolicExpression(OperandMixin):
    """Base class for symbolic expressions."""

    def replace(self, *mappers) -> SymbolicExpression:
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
        return _translators().replace(self, *mappers)

    def replace_by_name(self, **symbols) -> SymbolicExpression:
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
        return _translators().replace_by_name(self, **symbols)

    def eval(self, libsl: types.ModuleType = None):
        """Evaluate expression.

        If no implementation library is provided:
        1. 'libsl' will be looked up going back though the stack
           until is found.
        2. If still not found, the implementation using the python
           math module will be used (and a warning will be issued).

        Parameters
        ----------
        libsl
            implementation module
        """
        if libsl is None:
            frame = inspect.currentframe().f_back
            while frame:
                if "libsl" in frame.f_locals:
                    libsl = frame.f_locals["libsl"]
                    break
                frame = frame.f_back
            else:
                from .libimpl import math

                warnings.warn("No libsl provided, defaulting to 'math'.")
                libsl = math
        return _translators().evaluate(self, libsl)

    def __str__(self):
        return _translators().as_string(self)


@dataclasses.dataclass(frozen=True)
class Symbol(SymbolicExpression):
    """A user defined symbol."""

    name: str
    prefix: str = ""

    def __str__(self):
        return self.prefix + self.name


@dataclasses.dataclass(frozen=True)
class Function(Symbol):
    """A callable symbol."""

    arity: int = None

    def __call__(self, *args, **kwargs):
        return Call(self, args, kwargs)

    def format(self, *args, **kwargs):
        plain_args = args + tuple(f"{k}={v}" for k, v in kwargs.items())
        return f"{str(self)}({', '.join((str(v) for v in plain_args))})"


@dataclasses.dataclass(frozen=True)
class Call(SymbolicExpression):
    """A Function that has been called with certain arguments."""

    func: Function
    args: tuple
    kwargs_items: tuple | dict = ()

    def __post_init__(self):
        if isinstance(self.kwargs_items, dict):
            object.__setattr__(self, "kwargs_items", tuple(self.kwargs_items.items()))

    @functools.cached_property
    def kwargs(self):
        return dict(self.kwargs_items)

    def __str__(self):
        return self.func.format(*self.args, **self.kwargs)


@dataclasses.dataclass(frozen=True)
class Operator(Function):
    """Operators are functions that will be mapped as unary, binary or
    ternary expressions instead of calls.
    """

    fmt: str = ""

    @classmethod
    def from_operator(cls, op, s):
        arity = s.count("{}")
        return cls("op_" + op.__name__, LIBPREFIX, arity, s)

    def format(self, *args, **kwargs):
        if self.fmt and self.arity == len(args):
            return self.fmt.format(*args, *kwargs)
        return super().format(*args, **kwargs)

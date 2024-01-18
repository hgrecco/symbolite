"""
    symbolite.abstract.symbol
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Objects and functions for symbol operations.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from operator import attrgetter
from typing import Any, Callable, Generator, Mapping

from typing_extensions import Self

from ..core import (
    Unsupported,
    evaluate,
    substitute,
    substitute_by_name,
)


def filter_namespace(
    namespace: str | None = "", include_anonymous: bool = False
) -> Callable[[Named], bool]:
    def _inner(s: Named) -> bool:
        if namespace is None:
            return True
        return s.namespace == namespace

    return _inner


@dataclasses.dataclass(frozen=True)
class Named:
    name: str | None = None
    namespace: str = ""

    def __str__(self) -> str:
        if self.name is None:
            return "<anonymous>"

        if self.namespace:
            return self.namespace + "." + self.name

        return self.name

    @property
    def is_anonymous(self) -> bool:
        return self.name is None

    def format(self, *args: Any, **kwargs: Any) -> str:
        ...


@dataclasses.dataclass(frozen=True)
class Symbol(Named):
    """Base class for objects that might operate with others using
    python operators that map to magic methods

    The following magic methods are not mapped to symbolite Functions
      - __hash__, __eq__, __ne__ collides with reasonble use of comparisons
        within user code (including uses as dict keys).
        We defined `.eq` y `.ne` methods for the two lasts.
      - __contains__ is coerced to boolean.
      - __bool__ yields a TypeError if not boolean.
      - __str__, __bytes__, __repr__ yields a TypeError if the return value
        is not of the corresponding type.
        and they might also affect usability in the console.
      - __format__
      - __int__, __float__, __complex__ yields a TypeError if the return value
        is not of the corresponding type.
      - __round__, __abs__, __divmod__ they are too "numeric related"
      - __trunc__, __ceil__, __floor__ they are too "numeric related"
        and called by functions in math.
      - __len__ yields a TypeError if not int.
      - __index__ yields a TypeError if not int.

    Also, magic methods that are statements (not expressions) are also not
    mapped: e.g. __setitem__ or __delitem__

    """

    expression: Expression | None = None

    # Comparison methods (not operator)
    def eq(self, other: Any) -> Self:
        return eq(self, other)

    def ne(self, other: Any) -> Self:
        return ne(self, other)

    # Comparison magic methods
    def __lt__(self, other: Any) -> Self:
        """Implements less than comparison using the < operator."""
        return lt(self, other)

    def __le__(self, other: Any) -> Self:
        """Implements less than or equal comparison using the <= operator."""
        return le(self, other)

    def __gt__(self, other: Any) -> Self:
        """Implements greater than comparison using the > operator."""
        return gt(self, other)

    def __ge__(self, other: Any) -> Self:
        """Implements greater than or equal comparison using the >= operator."""
        return ge(self, other)

    # Emulating container types
    def __getitem__(self, key: Any) -> Self:
        """Defines behavior for when an item is accessed,
        using the notation self[key]."""
        return getitem(self, key)

    # Emulating attribute
    def __getattr__(self, key: str) -> Self:
        """Defines behavior for when an item is accessed,
        using the notation self.key"""
        if key.startswith("__"):
            raise AttributeError(key)
        return symgetattr(self, key)

    # Normal arithmetic operators
    def __add__(self, other: Any) -> Self:
        """Implements addition."""
        return add(self, other)

    def __sub__(self, other: Any) -> Self:
        """Implements subtraction."""
        return sub(self, other)

    def __mul__(self, other: Any) -> Self:
        """Implements multiplication."""
        return mul(self, other)

    def __matmul__(self, other: Any) -> Self:
        """Implements multiplication."""
        return matmul(self, other)

    def __truediv__(self, other: Any) -> Self:
        """Implements true division."""
        return truediv(self, other)

    def __floordiv__(self, other: Any) -> Self:
        """Implements integer division using the // operator."""
        return floordiv(self, other)

    def __mod__(self, other: Any) -> Self:
        """Implements modulo using the % operator."""
        return mod(self, other)

    def __pow__(self, other: Any, modulo: Any = None) -> Self:
        """Implements behavior for exponents using the ** operator."""
        if modulo is None:
            return pow(self, other)
        else:
            return pow3(self, other, modulo)

    def __lshift__(self, other: Any) -> Self:
        """Implements left bitwise shift using the << operator."""
        return lshift(self, other)

    def __rshift__(self, other: Any) -> Self:
        """Implements right bitwise shift using the >> operator."""
        return rshift(self, other)

    def __and__(self, other: Any) -> Self:
        """Implements bitwise and using the & operator."""
        return and_(self, other)

    def __or__(self, other: Any) -> Self:
        """Implements bitwise or using the | operator."""
        return or_(self, other)

    def __xor__(self, other: Any) -> Self:
        """Implements bitwise xor using the ^ operator."""
        return xor(self, other)

    # Reflected arithmetic operators
    def __radd__(self, other: Any) -> Self:
        """Implements reflected addition."""
        return add(other, self)

    def __rsub__(self, other: Any) -> Self:
        """Implements reflected subtraction."""
        return sub(other, self)

    def __rmul__(self, other: Any) -> Self:
        """Implements reflected multiplication."""
        return mul(other, self)

    def __rmatmul__(self, other: Any) -> Self:
        """Implements reflected multiplication."""
        return matmul(other, self)

    def __rtruediv__(self, other: Any) -> Self:
        """Implements reflected true division."""
        return truediv(other, self)

    def __rfloordiv__(self, other: Any) -> Self:
        """Implements reflected integer division using the // operator."""
        return floordiv(other, self)

    def __rmod__(self, other: Any) -> Self:
        """Implements reflected modulo using the % operator."""
        return mod(other, self)

    def __rpow__(self, other: Any) -> Self:
        """Implements behavior for reflected exponents using the ** operator."""
        return pow(other, self)

    def __rlshift__(self, other: Any) -> Self:
        """Implements reflected left bitwise shift using the << operator."""
        return lshift(other, self)

    def __rrshift__(self, other: Any) -> Self:
        """Implements reflected right bitwise shift using the >> operator."""
        return rshift(other, self)

    def __rand__(self, other: Any) -> Self:
        """Implements reflected bitwise and using the & operator."""
        return and_(other, self)

    def __ror__(self, other: Any) -> Self:
        """Implements reflected bitwise or using the | operator."""
        return or_(other, self)

    def __rxor__(self, other: Any) -> Self:
        """Implements reflected bitwise xor using the ^ operator."""
        return xor(other, self)

    # Unary operators and functions
    def __neg__(self) -> Self:
        """Implements behavior for negation (e.g. -some_object)"""
        return neg(self)

    def __pos__(self) -> Self:
        """Implements behavior for unary positive (e.g. +some_object)"""
        return pos(self)

    def __invert__(self) -> Self:
        """Implements behavior for inversion using the ~ operator."""
        return invert(self)

    def __str__(self) -> str:
        if self.expression is None:
            return super().__str__()
        return str(self.expression)

    def yield_named(
        self, include_anonymous: bool = False
    ) -> Generator[Named, None, None]:
        if self.expression is None:
            if include_anonymous or not self.is_anonymous:
                yield self
        else:
            yield from self.expression.yield_named(include_anonymous)

    def subs(self, mapper: Mapping[Any, Any]) -> Self:
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
        if self.expression is None:
            return mapper.get(self, self)
        return substitute(self.expression, mapper)

    def subs_by_name(self, **mapper: Any) -> Self:
        """Replace Symbols by values or objects, matching by name.

        If multiple mappers are provided,
            they will be used in order (using a ChainMap)

        If a given object is not found in the mappers,
            the same object will be returned.

        Parameters
        ----------
        **mapper
            keyword arguments connecting names to values.
        """
        if self.expression is None:
            return mapper.get(str(self), self)
        return substitute_by_name(self.expression, **mapper)

    def eval(self, libsl: types.ModuleType | None = None) -> Any:
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
        if libsl is None:
            return evaluate(self)

        if self.expression is not None:
            return evaluate(self.expression, libsl)

        if self.namespace:
            name = str(self)

            value = attrgetter(name)(libsl)

            if value is Unsupported:
                raise Unsupported(f"{name} is not supported in module {libsl.__name__}")

            return value
        else:
            # User defined symbol, txry to map the class
            name = (
                f"{self.__class__.__module__.split('.')[-1]}.{self.__class__.__name__}"
            )
            f = attrgetter(name)(libsl)

            if f is Unsupported:
                raise Unsupported(f"{name} is not supported in module {libsl.__name__}")

            return f(self.name)

    def symbol_namespaces(self) -> set[str]:
        """Return a set of symbol libraries"""
        return set(map(lambda s: s.namespace, self.yield_named(False)))

    def symbol_names(self, namespace: str | None = "") -> set[str]:
        """Return a set of symbol names (with full namespace indication).

        Parameters
        ----------
        namespace: str or None
            If None, all symbols will be returned independently of the namespace.
            If a string, will compare Symbol.namespace to that.
            Defaults to "" which is the namespace for user defined symbols.
        """
        ff = filter_namespace(namespace)
        return set(map(str, filter(ff, self.yield_named(False))))


@dataclasses.dataclass(frozen=True, kw_only=True)
class BaseFunction(Named):
    """A callable primitive that will return a call."""

    fmt: str | None = None
    arity: int | None = None

    @property
    def call(self) -> type[Expression]:
        return Expression

    @property
    def output_type(self):
        return Symbol

    def _call(self, *args: Any, **kwargs: Any) -> Symbol:
        return self.output_type(expression=self._build_resolver(*args, **kwargs))

    def _build_resolver(self, *args: Any, **kwargs: Any) -> Expression:
        if self.arity is None:
            return self.call(self, args, tuple(kwargs.items()))
        if kwargs:
            raise ValueError(
                "If arity is given, keyword arguments should not be provided."
            )
        if len(args) != self.arity:
            raise ValueError(
                f"Invalid number of arguments ({len(args)}), expected {self.arity}."
            )
        return self.call(self, args)

    def format(self, *args: Any, **kwargs: Any) -> str:
        if self.fmt:
            return self.fmt.format(*args, **kwargs)

        plain_args = args + tuple(f"{k}={v}" for k, v in kwargs.items())
        return f"{str(self)}({', '.join((str(v) for v in plain_args))})"


@dataclasses.dataclass(frozen=True)
class Function(BaseFunction):
    def __call__(self, *args: Any, **kwargs: Any) -> Symbol:
        return self._call(*args, **kwargs)


def _add_parenthesis(
    self: UnaryFunction | BinaryFunction,
    arg: UnaryFunction | BinaryFunction,
    *,
    right: bool,
) -> str:
    match arg:
        case Symbol(
            expression=Expression(
                func=UnaryFunction(precedence=p) | BinaryFunction(precedence=p)
            )
        ):
            if p < self.precedence or (right and p <= self.precedence):
                return f"({arg})"
    return str(arg)


@dataclasses.dataclass(frozen=True, kw_only=True)
class UnaryFunction(BaseFunction):
    arity: int = 1
    precedence: int

    def format(self, *args: Any, **kwargs: Any) -> str:
        (x,) = args
        x = _add_parenthesis(self, x, right=False)
        return super().format(x)

    def __call__(self, arg1: Symbol) -> Symbol:
        return self._call(arg1)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BinaryFunction(BaseFunction):
    arity: int = 2
    precedence: int

    def format(self, *args: Any, **kwargs: Any) -> str:
        x, y = args
        x = _add_parenthesis(self, x, right=False)
        y = _add_parenthesis(self, y, right=True)
        return super().format(x, y)

    def __call__(self, arg1: Symbol, arg2: Symbol) -> Symbol:
        return self._call(arg1, arg2)


@dataclasses.dataclass(frozen=True)
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

    def yield_named(
        self, include_anonymous: bool = False
    ) -> Generator[Named, None, None]:
        if include_anonymous or not self.func.is_anonymous:
            yield self.func

        for arg in self.args:
            if isinstance(arg, Symbol):
                yield from arg.yield_named(include_anonymous)

        for _, v in self.kwargs_items:
            if isinstance(v, Symbol):
                yield from v.yield_named(include_anonymous)

    def subs(self, mapper: Mapping[Any, Any]) -> Self:
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

        try:
            return func(*args, **kwargs)
        except Exception as ex:
            try:
                ex.add_note(f"While evaluating {func}(*{args}, **{kwargs}): {ex}")
            except AttributeError:
                pass
            raise ex

    def subs_by_name(self, **mapper: Any) -> Self:
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

        try:
            return func(*args, **kwargs)
        except Exception as ex:
            try:
                ex.add_note(f"While evaluating {func}(*{args}, **{kwargs}): {ex}")
            except AttributeError:
                pass
            raise ex

    def eval(self, libsl: types.ModuleType | None = None) -> Any:
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
        func = attrgetter(str(self.func))(libsl)
        args = tuple(evaluate(arg, libsl) for arg in self.args)
        kwargs = {k: evaluate(arg, libsl) for k, arg in self.kwargs_items}

        try:
            return func(*args, **kwargs)
        except Exception as ex:
            try:
                ex.add_note(f"While evaluating {func}(*{args}, **{kwargs}): {ex}")
            except AttributeError:
                pass
            raise ex

    def symbol_names(self, namespace: str | None = "") -> set[str]:
        """Return a set of symbol names (with full namespace indication).

        Parameters
        ----------
        namespace: str or None
            If None, all symbols will be returned independently of the namespace.
            If a string, will compare Symbol.namespace to that.
            Defaults to "" which is the namespace for user defined symbols.
        """
        ff = filter_namespace(namespace)
        return set(map(str, filter(ff, self.yield_named(False))))


# Comparison methods (not operator)
eq = BinaryFunction("eq", "symbol", precedence=-5, fmt="{} == {}")
ne = BinaryFunction("ne", "symbol", precedence=-5, fmt="{} != {}")

# Comparison
lt = BinaryFunction("lt", "symbol", precedence=-5, fmt="{} < {}")
le = BinaryFunction("le", "symbol", precedence=-5, fmt="{} <= {}")
gt = BinaryFunction("gt", "symbol", precedence=-5, fmt="{} > {}")
ge = BinaryFunction("ge", "symbol", precedence=-5, fmt="{} >= {}")

# Emulating container types
getitem = BinaryFunction("getitem", "symbol", precedence=5, fmt="{}[{}]")

# Emulating attribute
symgetattr = BinaryFunction("symgetattr", "symbol", precedence=5, fmt="{}.{}")

# Emulating numeric types
add = BinaryFunction("add", "symbol", precedence=0, fmt="{} + {}")
sub = BinaryFunction("sub", "symbol", precedence=0, fmt="{} - {}")
mul = BinaryFunction("mul", "symbol", precedence=1, fmt="{} * {}")
matmul = BinaryFunction("matmul", "symbol", precedence=1, fmt="{} @ {}")
truediv = BinaryFunction("truediv", "symbol", precedence=1, fmt="{} / {}")
floordiv = BinaryFunction("floordiv", "symbol", precedence=1, fmt="{} // {}")
mod = BinaryFunction("mod", "symbol", precedence=1, fmt="{} % {}")
pow = BinaryFunction("pow", "symbol", precedence=3, fmt="{} ** {}")
pow3 = Function("pow3", "symbol", fmt="pow({}, {}, {})", arity=3)
lshift = BinaryFunction("lshift", "symbol", precedence=-1, fmt="{} << {}")
rshift = BinaryFunction("rshift", "symbol", precedence=-1, fmt="{} >> {}")
and_ = BinaryFunction("and_", "symbol", precedence=-2, fmt="{} & {}")
xor = BinaryFunction("xor", "symbol", precedence=-3, fmt="{} ^ {}")
or_ = BinaryFunction("or_", "symbol", precedence=-4, fmt="{} | {}")

# Unary operators
neg = UnaryFunction("neg", "symbol", precedence=2, fmt="-{}")
pos = UnaryFunction("pos", "symbol", precedence=2, fmt="+{}")
invert = UnaryFunction("invert", "symbol", precedence=2, fmt="~{}")

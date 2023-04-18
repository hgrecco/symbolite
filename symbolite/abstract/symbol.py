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
import typing as ty

from ..core import as_string, evaluate, inspect, substitute, substitute_by_name
from ..core.base import BaseCall, BaseFunction, BaseSymbol, Named

Self = ty.TypeVar("Self")


@dataclasses.dataclass(frozen=True)
class Function(Named, BaseFunction):
    """A callable primitive that will return a call."""

    arity: int | None = None
    fmt: str | None = None

    @property
    def call(self) -> type[Call]:
        return Call

    def __call__(self, *args: ty.Any, **kwargs: ty.Any):
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

    def format(self, *args: ty.Any, **kwargs: ty.Any):
        if self.fmt:
            return self.fmt.format(*args, **kwargs)

        plain_args = args + tuple(f"{k}={v}" for k, v in kwargs.items())
        return f"{str(self)}({', '.join((str(v) for v in plain_args))})"

    def impl(self: Self, impl_name: str, func: ty.Callable[..., ty.Any]) -> Self:
        self.impls[impl_name] = func  # type: ignore
        return self


@dataclasses.dataclass(frozen=True)
class OperandMixin:
    """Base class for objects that might operate with others using
    python operators that map to magic methods

    The following magic methods are not mapped to symbolite Functions
      - __eq__, __hash__, __ne__ collides with reasonble use of comparisons
        within user code (including uses as dict keys).
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

    # Comparison methods (not operator)
    def eq(self, other: ty.Any) -> Call:
        return eq(self, other)

    def ne(self, other: ty.Any) -> Call:
        return eq(self, other)

    # Comparison magic methods
    def __lt__(self, other: ty.Any) -> Call:
        """Implements less than comparison using the < operator."""
        return lt(self, other)

    def __le__(self, other: ty.Any) -> Call:
        """Implements less than or equal comparison using the <= operator."""
        return le(self, other)

    def __gt__(self, other: ty.Any) -> Call:
        """Implements greater than comparison using the > operator."""
        return gt(self, other)

    def __ge__(self, other: ty.Any) -> Call:
        """Implements greater than or equal comparison using the >= operator."""
        return ge(self, other)

    # Emulating container types
    def __getitem__(self, key: ty.Any) -> Call:
        """Defines behavior for when an item is accessed,
        using the notation self[key]."""
        return getitem(self, key)

    # Normal arithmetic operators
    def __add__(self, other: ty.Any) -> Call:
        """Implements addition."""
        return add(self, other)

    def __sub__(self, other: ty.Any) -> Call:
        """Implements subtraction."""
        return sub(self, other)

    def __mul__(self, other: ty.Any) -> Call:
        """Implements multiplication."""
        return mul(self, other)

    def __matmul__(self, other: ty.Any) -> Call:
        """Implements multiplication."""
        return matmul(self, other)

    def __truediv__(self, other: ty.Any) -> Call:
        """Implements true division."""
        return truediv(self, other)

    def __floordiv__(self, other: ty.Any) -> Call:
        """Implements integer division using the // operator."""
        return floordiv(self, other)

    def __mod__(self, other: ty.Any) -> Call:
        """Implements modulo using the % operator."""
        return mod(self, other)

    def __pow__(self, other: ty.Any, modulo: ty.Any = None) -> Call:
        """Implements behavior for exponents using the ** operator."""
        if modulo is None:
            return pow(self, other)
        else:
            return pow3(self, other, modulo)

    def __lshift__(self, other: ty.Any) -> Call:
        """Implements left bitwise shift using the << operator."""
        return lshift(self, other)

    def __rshift__(self, other: ty.Any) -> Call:
        """Implements right bitwise shift using the >> operator."""
        return rshift(self, other)

    def __and__(self, other: ty.Any) -> Call:
        """Implements bitwise and using the & operator."""
        return and_(self, other)

    def __or__(self, other: ty.Any) -> Call:
        """Implements bitwise or using the | operator."""
        return or_(self, other)

    def __xor__(self, other: ty.Any) -> Call:
        """Implements bitwise xor using the ^ operator."""
        return xor(self, other)

    # Reflected arithmetic operators
    def __radd__(self, other: ty.Any) -> Call:
        """Implements reflected addition."""
        return radd(self, other)

    def __rsub__(self, other: ty.Any) -> Call:
        """Implements reflected subtraction."""
        return rsub(self, other)

    def __rmul__(self, other: ty.Any) -> Call:
        """Implements reflected multiplication."""
        return rmul(self, other)

    def __rmatmul__(self, other: ty.Any) -> Call:
        """Implements reflected multiplication."""
        return rmatmul(self, other)

    def __rtruediv__(self, other: ty.Any) -> Call:
        """Implements reflected true division."""
        return rtruediv(self, other)

    def __rfloordiv__(self, other: ty.Any) -> Call:
        """Implements reflected integer division using the // operator."""
        return rfloordiv(self, other)

    def __rmod__(self, other: ty.Any) -> Call:
        """Implements reflected modulo using the % operator."""
        return rmod(self, other)

    def __rpow__(self, other: ty.Any) -> Call:
        """Implements behavior for reflected exponents using the ** operator."""
        return rpow(self, other)

    def __rlshift__(self, other: ty.Any) -> Call:
        """Implements reflected left bitwise shift using the << operator."""
        return rlshift(self, other)

    def __rrshift__(self, other: ty.Any) -> Call:
        """Implements reflected right bitwise shift using the >> operator."""
        return rrshift(self, other)

    def __rand__(self, other: ty.Any) -> Call:
        """Implements reflected bitwise and using the & operator."""
        return rand(self, other)

    def __ror__(self, other: ty.Any) -> Call:
        """Implements reflected bitwise or using the | operator."""
        return ror(self, other)

    def __rxor__(self, other: ty.Any) -> Call:
        """Implements reflected bitwise xor using the ^ operator."""
        return rxor(self, other)

    # Unary operators and functions
    def __neg__(self) -> Call:
        """Implements behavior for negation (e.g. -some_object)"""
        return neg(self)

    def __pos__(self) -> Call:
        """Implements behavior for unary positive (e.g. +some_object)"""
        return pos(self)

    def __invert__(self) -> Call:
        """Implements behavior for inversion using the ~ operator."""
        return invert(self)

    def subs(self, *mappers: dict[ty.Any, ty.Any]) -> OperandMixin:
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
        return substitute(self, *mappers)

    def subs_by_name(self, **symbols: ty.Any) -> OperandMixin:
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
        return substitute_by_name(self, **symbols)

    def eval(self, libsl: types.ModuleType | None = None) -> ty.Any:
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

        return evaluate(self, libsl)

    def symbol_namespaces(self) -> set[str]:
        """Return a set of symbol libraries"""
        symbols = (s for s in inspect(self) if isinstance(s, Named))
        return set(map(lambda s: s.namespace, symbols))

    def symbol_names(self, namespace: str | None = "") -> set[str]:
        """Return a set of symbol names (with full namespace indication).

        Parameters
        ----------
        namespace: str or None
            If None, all symbols will be returned independently of the namespace.
            If a string, will compare Symbol.namespace to that.
            Defaults to "" which is the namespace for user defined symbols.
        """
        symbols = (s for s in inspect(self) if isinstance(s, Named))

        namespaces: list[str] = []
        if namespace is not None:
            namespaces.append(namespace)

        if namespaces:
            symbols = (s for s in symbols if s.namespace in namespaces)

        return set(map(str, symbols))

    def __str__(self):
        return as_string(self)


@dataclasses.dataclass(frozen=True)
class Call(OperandMixin, BaseCall):
    """A Function that has been called with certain arguments."""

    func: Function
    args: tuple[ty.Any]
    kwargs_items: tuple[tuple[str, ty.Any], ...] = ()

    def __post_init__(self):
        if isinstance(self.kwargs_items, dict):
            object.__setattr__(self, "kwargs_items", tuple(self.kwargs_items.items()))

    @functools.cached_property
    def kwargs(self):
        return dict(self.kwargs_items)

    def __str__(self):
        return self.func.format(*self.args, *self.kwargs)


@dataclasses.dataclass(frozen=True)
class Symbol(Named, OperandMixin, BaseSymbol):
    """A symbol."""

    namespace = ""


@dataclasses.dataclass(frozen=True)
class SymbolFunction(Function):
    namespace = "symbol"


# Comparison methods (not operator)
eq = SymbolFunction("eq", 2, "({} == {})")
ne = SymbolFunction("ne", 2, "({} != {})")

# Comparison
lt = SymbolFunction("lt", 2, "({} < {})")
le = SymbolFunction("le", 2, "({} <= {})")
gt = SymbolFunction("gt", 2, "({} > {})")
ge = SymbolFunction("ge", 2, "({} >= {})")

# Emulating container types
getitem = SymbolFunction("getitem", 2, "{}[{}]")

# Emulating numeric types
add = SymbolFunction("add", 2, "({} + {})")
sub = SymbolFunction("sub", 2, "({} - {})")
mul = SymbolFunction("mul", 2, "({} * {})")
matmul = SymbolFunction("matmul", 2, "({} @ {})")
truediv = SymbolFunction("truediv", 2, "({} / {})")
floordiv = SymbolFunction("floordiv", 2, "({} // {})")
mod = SymbolFunction("mod", 2, "({} % {})")
pow = SymbolFunction("pow", 2, "({} ** {})")
pow3 = SymbolFunction("pow3", 3, "pow({}, {}, {})")
lshift = SymbolFunction("lshift", 2, "({} << {})")
rshift = SymbolFunction("rshift", 2, "({} >> {})")
and_ = SymbolFunction("and_", 2, "({} & {})")
xor = SymbolFunction("xor", 2, "({} ^ {})")
or_ = SymbolFunction("or_", 2, "({} | {})")

# Reflective versions
radd = SymbolFunction("radd", 2, "({1} + {0})")
rsub = SymbolFunction("rsub", 2, "({1} - {0})")
rmul = SymbolFunction("rmul", 2, "({1} * {0})")
rmatmul = SymbolFunction("rmatmul", 2, "({1} @ {0})")
rtruediv = SymbolFunction("rtruediv", 2, "({1} / {0})")
rfloordiv = SymbolFunction("rfloordiv", 2, "({1} // {0})")
rmod = SymbolFunction("rmod", 2, "({1} % {0})")
rpow = SymbolFunction("pow", 2, "({1} ** {0})")
rlshift = SymbolFunction("rlshift", 2, "({1} << {0})")
rrshift = SymbolFunction("rrshift", 2, "({1} >> {0})")
rand = SymbolFunction("rand", 2, "({1} & {0})")
rxor = SymbolFunction("rxor", 2, "({1} ^ {0})")
ror = SymbolFunction("ror", 2, "({1} | {0})")

# Reflective versions
neg = SymbolFunction("neg", 1, "(-{})")
pos = SymbolFunction("pos", 1, "(+{})")
invert = SymbolFunction("invert", 1, "(~{})")

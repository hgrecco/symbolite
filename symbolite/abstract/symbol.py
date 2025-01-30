"""
symbolite.abstract.symbol
~~~~~~~~~~~~~~~~~~~~~~~~~

Objects and functions for symbol operations.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import types
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Literal,
    Mapping,
    ParamSpec,
    TypeVar,
)

from typing_extensions import Self

from ..core.expression import Expression, NamedExpression
from ..core.function import BaseFunction
from ..core.named import Named, yield_named
from ..core.operations import (
    evaluate_impl,
    substitute,
    substitute_by_name,
)
from ..core.util import Unsupported, repr_without_defaults

P = ParamSpec("P")
T = TypeVar("T")


@dataclasses.dataclass(frozen=True, repr=False)
class Symbol(NamedExpression):
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

    # Comparison methods (not operator)
    def eq(self, other: Any) -> Symbol:
        return eq(self, other)

    def ne(self, other: Any) -> Symbol:
        return ne(self, other)

    # Comparison magic methods
    def __lt__(self, other: Any) -> Symbol:
        """Implements less than comparison using the < operator."""
        return lt(self, other)

    def __le__(self, other: Any) -> Symbol:
        """Implements less than or equal comparison using the <= operator."""
        return le(self, other)

    def __gt__(self, other: Any) -> Symbol:
        """Implements greater than comparison using the > operator."""
        return gt(self, other)

    def __ge__(self, other: Any) -> Symbol:
        """Implements greater than or equal comparison using the >= operator."""
        return ge(self, other)

    # Emulating container types
    def __getitem__(self, key: Any) -> Symbol:
        """Defines behavior for when an item is accessed,
        using the notation self[key]."""
        return getitem(self, key)

    # Emulating attribute
    def __getattr__(self, key: str) -> Symbol:
        """Defines behavior for when an item is accessed,
        using the notation self.key"""
        if key.startswith("__"):
            raise AttributeError(key)
        return symgetattr(self, key)

    # Normal arithmetic operators
    def __add__(self, other: Any) -> Symbol:
        """Implements addition."""
        return add(self, other)

    def __sub__(self, other: Any) -> Symbol:
        """Implements subtraction."""
        return sub(self, other)

    def __mul__(self, other: Any) -> Symbol:
        """Implements multiplication."""
        return mul(self, other)

    def __matmul__(self, other: Any) -> Symbol:
        """Implements multiplication."""
        return matmul(self, other)

    def __truediv__(self, other: Any) -> Symbol:
        """Implements true division."""
        return truediv(self, other)

    def __floordiv__(self, other: Any) -> Symbol:
        """Implements integer division using the // operator."""
        return floordiv(self, other)

    def __mod__(self, other: Any) -> Symbol:
        """Implements modulo using the % operator."""
        return mod(self, other)

    def __pow__(self, other: Any, modulo: Any = None) -> Symbol:
        """Implements behavior for exponents using the ** operator."""
        if modulo is None:
            return pow(self, other)
        else:
            return pow3(self, other, modulo)

    def __lshift__(self, other: Any) -> Symbol:
        """Implements left bitwise shift using the << operator."""
        return lshift(self, other)

    def __rshift__(self, other: Any) -> Symbol:
        """Implements right bitwise shift using the >> operator."""
        return rshift(self, other)

    def __and__(self, other: Any) -> Symbol:
        """Implements bitwise and using the & operator."""
        return and_(self, other)

    def __or__(self, other: Any) -> Symbol:
        """Implements bitwise or using the | operator."""
        return or_(self, other)

    def __xor__(self, other: Any) -> Symbol:
        """Implements bitwise xor using the ^ operator."""
        return xor(self, other)

    # Reflected arithmetic operators
    def __radd__(self, other: Any) -> Symbol:
        """Implements reflected addition."""
        return add(other, self)

    def __rsub__(self, other: Any) -> Symbol:
        """Implements reflected subtraction."""
        return sub(other, self)

    def __rmul__(self, other: Any) -> Symbol:
        """Implements reflected multiplication."""
        return mul(other, self)

    def __rmatmul__(self, other: Any) -> Symbol:
        """Implements reflected multiplication."""
        return matmul(other, self)

    def __rtruediv__(self, other: Any) -> Symbol:
        """Implements reflected true division."""
        return truediv(other, self)

    def __rfloordiv__(self, other: Any) -> Symbol:
        """Implements reflected integer division using the // operator."""
        return floordiv(other, self)

    def __rmod__(self, other: Any) -> Symbol:
        """Implements reflected modulo using the % operator."""
        return mod(other, self)

    def __rpow__(self, other: Any) -> Symbol:
        """Implements behavior for reflected exponents using the ** operator."""
        return pow(other, self)

    def __rlshift__(self, other: Any) -> Symbol:
        """Implements reflected left bitwise shift using the << operator."""
        return lshift(other, self)

    def __rrshift__(self, other: Any) -> Symbol:
        """Implements reflected right bitwise shift using the >> operator."""
        return rshift(other, self)

    def __rand__(self, other: Any) -> Symbol:
        """Implements reflected bitwise and using the & operator."""
        return and_(other, self)

    def __ror__(self, other: Any) -> Symbol:
        """Implements reflected bitwise or using the | operator."""
        return or_(other, self)

    def __rxor__(self, other: Any) -> Symbol:
        """Implements reflected bitwise xor using the ^ operator."""
        return xor(other, self)

    # Unary operators and functions
    def __neg__(self) -> Symbol:
        """Implements behavior for negation (e.g. -some_object)"""
        return neg(self)

    def __pos__(self) -> Symbol:
        """Implements behavior for unary positive (e.g. +some_object)"""
        return pos(self)

    def __invert__(self) -> Symbol:
        """Implements behavior for inversion using the ~ operator."""
        return invert(self)

    def __str__(self) -> str:
        if self.expression is None:
            return super().__str__()
        return str(self.expression)

    # Naming in symbolic namespace
    def __set_name__(self, owner: Any, name: str):
        object.__setattr__(self, "name", name)


@yield_named.register
def _(self: Symbol, include_anonymous: bool = False) -> Generator[Named, None, None]:
    if self.expression is None:
        if include_anonymous or not self.is_anonymous:
            yield self
    else:
        yield from yield_named(self.expression, include_anonymous)


@substitute.register
def _(self: Symbol, mapper: Mapping[Any, Any]) -> Symbol:
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
    out = substitute(self.expression, mapper)
    if not isinstance(out, Expression):
        return out
    return self.__class__(name=self.name, namespace=self.namespace, expression=out)


@substitute_by_name.register
def _(self: Symbol, **mapper: Any) -> Symbol:
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
    out = substitute_by_name(self.expression, **mapper)
    if not isinstance(out, Expression):
        return out
    return self.__class__(name=self.name, namespace=self.namespace, expression=out)


@evaluate_impl.register
def _(self: Symbol, libsl: types.ModuleType) -> Any:
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

    if self.expression is not None:
        return evaluate_impl(self.expression, libsl)

    if self.namespace:
        name = str(self)

        value = evaluate_impl(name, libsl)

        if value is Unsupported:
            raise Unsupported(f"{name} is not supported in module {libsl.__name__}")

        return value
    else:
        # User defined symbol, txry to map the class
        name = f"{self.__class__.__module__.split('.')[-1]}.{self.__class__.__name__}"
        f = evaluate_impl(name, libsl)

        if f is Unsupported:
            raise Unsupported(f"{name} is not supported in module {libsl.__name__}")

        return f(self.name)


S = TypeVar("S", bound=Symbol)


def downcast(symbol_obj: Symbol, subclass: type[S]) -> S:
    return subclass(
        name=symbol_obj.name,
        namespace=symbol_obj.namespace,
        expression=symbol_obj.expression,
    )


def _add_parenthesis(
    self: UnaryFunction | BinaryFunction,
    arg: UnaryFunction | BinaryFunction | Symbol,
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
        case _:
            pass
    return str(arg)


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class Function(BaseFunction):
    @property
    def output_type(self) -> type[Symbol]:
        return Symbol

    def __call__(self, *args: Any, **kwargs: Any) -> Symbol:
        return self._call(*args, **kwargs)


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class UserFunction(Function, Generic[P, T]):
    _impls: dict[types.ModuleType | Literal["default"], Callable[P, T]] = (
        dataclasses.field(init=False, default_factory=dict)
    )

    @classmethod
    def from_function(cls, func: Callable[P, T]) -> Self:
        obj = cls(name=func.__name__, namespace="user")
        obj._impls["default"] = func
        return obj

    def __repr__(self) -> str:
        return repr_without_defaults(self, include_private=False)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Symbol:
        return super().__call__(*args, **kwargs)

    def register_impl(self, func: Callable[P, T], libsl: types.ModuleType):
        self._impls[libsl] = func


@evaluate_impl.register
def _(self: UserFunction, libsl: types.ModuleType) -> Callable[..., Any]:
    impls = self._impls
    if libsl in impls:
        return impls[libsl]
    elif "default" in impls:
        return impls["default"]
    else:
        raise Exception(
            f"No implementation found for {libsl.__name__} and no default implementation provided for function {self!s}"
        )


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class UnaryFunction(Function):
    arity: int = 1
    precedence: int

    def format(self, *args: Any, **kwargs: Any) -> str:
        (x,) = args
        x = _add_parenthesis(self, x, right=False)
        return super().format(x)

    def __call__(self, arg1: Symbol) -> Symbol:
        return self._call(arg1)


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class BinaryFunction(Function):
    arity: int = 2
    precedence: int

    @property
    def output_type(self) -> type[Symbol]:
        return Symbol

    def format(self, *args: Any, **kwargs: Any) -> str:
        x, y = args
        x = _add_parenthesis(self, x, right=False)
        y = _add_parenthesis(self, y, right=True)
        return super().format(x, y)

    def __call__(self, arg1: Symbol, arg2: Symbol) -> Symbol:
        return self._call(arg1, arg2)


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

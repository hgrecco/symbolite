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

from symbolite.core.util import repr_without_defaults

from ..core import (
    Unsupported,
    evaluate,
    evaluate_impl,
    substitute,
    substitute_by_name,
)

P = ParamSpec("P")
T = TypeVar("T")


def filter_namespace(
    namespace: str | None = "", include_anonymous: bool = False
) -> Callable[[Named], bool]:
    def _inner(s: Named) -> bool:
        if namespace is None:
            return True
        return s.namespace == namespace

    return _inner


@dataclasses.dataclass(frozen=True, repr=False)
class Named:
    name: str | None = None
    namespace: str = ""

    def __str__(self) -> str:
        if self.name is None:
            return "<anonymous>"

        if self.namespace:
            return self.namespace + "." + self.name

        return self.name

    def __repr__(self) -> str:
        return repr_without_defaults(self)

    @property
    def is_anonymous(self) -> bool:
        return self.name is None

    def format(self, *args: Any, **kwargs: Any) -> str: ...


def symbol_namespaces(self: Any) -> set[str]:
    """Return a set of symbol libraries"""
    return set(map(lambda s: s.namespace, yield_named(self, False)))


def symbol_names(self: Any, namespace: str | None = "") -> set[str]:
    """Return a set of symbol names (with full namespace indication).

    Parameters
    ----------
    namespace: str or None
        If None, all symbols will be returned independently of the namespace.
        If a string, will compare Symbol.namespace to that.
        Defaults to "" which is the namespace for user defined symbols.
    """
    ff = filter_namespace(namespace)
    return set(map(str, filter(ff, yield_named(self, False))))


@dataclasses.dataclass(frozen=True, repr=False)
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

    ##################################################
    # TODO: This will be deprecated in future versions.
    ##################################################

    def yield_named(
        self, include_anonymous: bool = False
    ) -> Generator[Named, None, None]:
        yield from yield_named(self, include_anonymous)

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
        return substitute(self, mapper)

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
        return substitute_by_name(self, **mapper)

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
        return evaluate(self, libsl)

    def symbol_namespaces(self) -> set[str]:
        """Return a set of symbol libraries"""
        return symbol_namespaces(self)

    def symbol_names(self, namespace: str | None = "") -> set[str]:
        """Return a set of symbol names (with full namespace indication).

        Parameters
        ----------
        namespace: str or None
            If None, all symbols will be returned independently of the namespace.
            If a string, will compare Symbol.namespace to that.
            Defaults to "" which is the namespace for user defined symbols.
        """
        return symbol_names(self, namespace)


@functools.singledispatch
def yield_named(
    self: Symbol, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    if self.expression is None:
        if include_anonymous or not self.is_anonymous:
            yield self
    else:
        yield from yield_named(self.expression, include_anonymous)


@substitute.register
def substitute_symbol(self: Symbol, mapper: Mapping[Any, Any]) -> Symbol:
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
def substitute_by_name_symbol(self: Symbol, **mapper: Any) -> Symbol:
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
def evaluate_impl_symbol(self: Symbol, libsl: types.ModuleType) -> Any:
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


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
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


@dataclasses.dataclass(frozen=True, repr=False)
class Function(BaseFunction):
    def __call__(self, *args: Any, **kwargs: Any) -> Symbol:
        return self._call(*args, **kwargs)


@evaluate_impl.register
def evaluate_impl_function(
    expr: BaseFunction, libsl: types.ModuleType
) -> Any | Unsupported:
    return attrgetter(str(expr))(libsl)


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
def evaluate_impl_user_function(
    self: UserFunction, libsl: types.ModuleType
) -> Callable[P, T]:
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
class UnaryFunction(BaseFunction):
    arity: int = 1
    precedence: int

    def format(self, *args: Any, **kwargs: Any) -> str:
        (x,) = args
        x = _add_parenthesis(self, x, right=False)
        return super().format(x)

    def __call__(self, arg1: Symbol) -> Symbol:
        return self._call(arg1)


@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
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

    ##################################################
    # TODO: This will be deprecated in future versions.
    ##################################################

    def yield_named(
        self, include_anonymous: bool = False
    ) -> Generator[Named, None, None]:
        yield from yield_named(self, include_anonymous)

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
        return substitute(self, mapper)

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
        return substitute_by_name(self, **mapper)

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
        return evaluate(self, libsl)

    def symbol_names(self, namespace: str | None = "") -> set[str]:
        """Return a set of symbol names (with full namespace indication).

        Parameters
        ----------
        namespace: str or None
            If None, all symbols will be returned independently of the namespace.
            If a string, will compare Symbol.namespace to that.
            Defaults to "" which is the namespace for user defined symbols.
        """
        return symbol_names(self)


@yield_named.register
def yield_named_expression(
    self: Expression, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    if include_anonymous or not self.func.is_anonymous:
        yield self.func

    for arg in self.args:
        if isinstance(arg, Symbol):
            yield from yield_named(arg, include_anonymous)

    for _, v in self.kwargs_items:
        if isinstance(v, Symbol):
            yield from yield_named(v, include_anonymous)


@substitute.register
def substitute_expression(self: Expression, mapper: Mapping[Any, Any]) -> Expression:
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
def substitute_by_name_expression(self: Expression, **mapper: Any) -> Expression:
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
def evaluate_impl_expression(self: Expression, libsl: types.ModuleType) -> Any:
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
    args = tuple(evaluate(arg, libsl) for arg in self.args)
    kwargs = {k: evaluate_impl(arg, libsl) for k, arg in self.kwargs_items}

    try:
        return func(*args, **kwargs)
    except Exception as ex:
        try:
            ex.add_note(f"While evaluating {func}(*{args}, **{kwargs}): {ex}")
        except AttributeError:
            pass
        raise ex


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

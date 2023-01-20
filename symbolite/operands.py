"""
    symbolite.operands
    ~~~~~~~~~~~~~~~~~~

    Expression operands.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import functools
import types
import typing

dataclass = dataclasses.dataclass(frozen=True)  # , eq=False, order=False)


@functools.lru_cache()
def _translators():
    # Is there a better way to deal with circular imports.
    from . import translators

    return translators


@functools.lru_cache()
def _datamodel():
    # Is there a better way to deal with circular imports.
    from . import datamodel

    return datamodel


@dataclass
class Named:
    """A named primitive."""

    name: str
    namespace: str = ""

    def __str__(self):
        if self.namespace:
            return self.namespace + "." + self.name
        return self.name


@dataclass
class Function(Named):
    """A callable primitive that will return a call."""

    arity: int = None
    fmt: str = None

    def __call__(self, *args, **kwargs):
        if self.arity is None:
            return Call(self, args, tuple(kwargs.items()))
        if kwargs:
            raise ValueError(
                "If arity is given, keyword arguments should not be provided."
            )
        if len(args) != self.arity:
            raise ValueError(
                f"Invalid number of arguments ({len(args)}), expected {self.arity}."
            )
        return Call(self, args)

    def format(self, *args, **kwargs):
        if self.fmt:
            return self.fmt.format(*args, **kwargs)

        plain_args = args + tuple(f"{k}={v}" for k, v in kwargs.items())
        return f"{str(self)}({', '.join((str(v) for v in plain_args))})"


@dataclass
class OperandMixin:
    """Base class for objects that might operate with others using
    python operators that map to magic methods

    The following magic methods are not mapped to symbolite Functions
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

    # Comparison magic methods
    def __lt__(self, other):
        """Implements less than comparison using the < operator."""
        return Call(_datamodel().__lt__, (self, other))

    def __le__(self, other):
        """Implements less than or equal comparison using the <= operator."""
        return Call(_datamodel().__le__, (self, other))

    def __hash__(self):
        return hash(self)

    def __eq__(self, other):
        """Implements equal comparison using the == operator."""
        return Call(_datamodel().__eq__, (self, other))

    def __ne__(self, other):
        """Implements not equal comparison using the != operator."""
        return Call(_datamodel().__ne__, (self, other))

    def __gt__(self, other):
        """Implements greater than comparison using the > operator."""
        return Call(_datamodel().__gt__, (self, other))

    def __ge__(self, other):
        """Implements greater than or equal comparison using the >= operator."""
        return Call(_datamodel().__ge__, (self, other))

    # Emulating container types
    def __getitem__(self, key):
        """Defines behavior for when an item is accessed,
        using the notation self[key]."""
        return Call(_datamodel().__getitem__, (self, key))

    # Normal arithmetic operators

    def __add__(self, other) -> Call:
        """Implements addition."""
        return Call(_datamodel().__add__, (self, other))

    def __sub__(self, other) -> Call:
        """Implements subtraction."""
        return Call(_datamodel().__sub__, (self, other))

    def __mul__(self, other) -> Call:
        """Implements multiplication."""
        return Call(_datamodel().__mul__, (self, other))

    def __matmul__(self, other) -> Call:
        """Implements multiplication."""
        return Call(_datamodel().__matmul__, (self, other))

    def __truediv__(self, other) -> Call:
        """Implements true division."""
        return Call(_datamodel().__truediv__, (self, other))

    def __floordiv__(self, other) -> Call:
        """Implements integer division using the // operator."""
        return Call(_datamodel().__floordiv__, (self, other))

    def __mod__(self, other) -> Call:
        """Implements modulo using the % operator."""
        return Call(_datamodel().__mod__, (self, other))

    def __pow__(self, other, modulo=None) -> Call:
        """Implements behavior for exponents using the ** operator."""
        if modulo is None:
            return Call(_datamodel().__pow__, (self, other))
        else:
            return Call(_datamodel().__pow3__, (self, other, modulo))

    def __lshift__(self, other) -> Call:
        """Implements left bitwise shift using the << operator."""
        return Call(_datamodel().__lshift__, (self, other))

    def __rshift__(self, other) -> Call:
        """Implements right bitwise shift using the >> operator."""
        return Call(_datamodel().__rshift__, (self, other))

    def __and__(self, other) -> Call:
        """Implements bitwise and using the & operator."""
        return Call(_datamodel().__and__, (self, other))

    def __or__(self, other) -> Call:
        """Implements bitwise or using the | operator."""
        return Call(_datamodel().__or__, (self, other))

    def __xor__(self, other) -> Call:
        """Implements bitwise xor using the ^ operator."""
        return Call(_datamodel().__xor__, (self, other))

    # Reflected arithmetic operators
    def __radd__(self, other) -> Call:
        """Implements reflected addition."""
        return Call(_datamodel().__radd__, (self, other))

    def __rsub__(self, other) -> Call:
        """Implements reflected subtraction."""
        return Call(_datamodel().__rsub__, (self, other))

    def __rmul__(self, other) -> Call:
        """Implements reflected multiplication."""
        return Call(_datamodel().__rmul__, (self, other))

    def __rmatmul__(self, other) -> Call:
        """Implements reflected multiplication."""
        return Call(_datamodel().__rmatmul__, (self, other))

    def __rtruediv__(self, other) -> Call:
        """Implements reflected true division."""
        return Call(_datamodel().__rtruediv__, (self, other))

    def __rfloordiv__(self, other) -> Call:
        """Implements reflected integer division using the // operator."""
        return Call(_datamodel().__rfloordiv__, (self, other))

    def __rmod__(self, other) -> Call:
        """Implements reflected modulo using the % operator."""
        return Call(_datamodel().__rmod__, (self, other))

    def __rpow__(self, other) -> Call:
        """Implements behavior for reflected exponents using the ** operator."""
        return Call(_datamodel().__rpow__, (self, other))

    def __rlshift__(self, other) -> Call:
        """Implements reflected left bitwise shift using the << operator."""
        return Call(_datamodel().__rlshift__, (self, other))

    def __rrshift__(self, other) -> Call:
        """Implements reflected right bitwise shift using the >> operator."""
        return Call(_datamodel().__rrshift__, (self, other))

    def __rand__(self, other) -> Call:
        """Implements reflected bitwise and using the & operator."""
        return Call(_datamodel().__rand__, (self, other))

    def __ror__(self, other) -> Call:
        """Implements reflected bitwise or using the | operator."""
        return Call(_datamodel().__ror__, (self, other))

    def __rxor__(self, other) -> Call:
        """Implements reflected bitwise xor using the ^ operator."""
        return Call(_datamodel().__rxor__, (self, other))

    # Unary operators and functions
    def __neg__(self) -> Call:
        """Implements behavior for negation (e.g. -some_object)"""
        return Call(_datamodel().__neg__, (self,))

    def __pos__(self) -> Call:
        """Implements behavior for unary positive (e.g. +some_object)"""
        return Call(_datamodel().__pos__, (self,))

    def __invert__(self) -> Call:
        """Implements behavior for inversion using the ~ operator."""
        return Call(_datamodel().__invert__, (self,))

    def subs(self, *mappers) -> OperandMixin:
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

    def subs_by_name(self, **symbols) -> OperandMixin:
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

    def eval(self, **libs: types.ModuleType) -> typing.Any:
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

        return _translators().evaluate(self, **libs)

    def symbol_namespaces(self) -> set[str, ...]:
        """Return a set of symbol libraries"""
        symbols = (s for s in _translators().inspect(self) if isinstance(s, Named))
        return set(map(lambda s: s.namespace, symbols))

    def symbol_names(self, namespace="") -> set[str, ...]:
        """Return a set of symbol names (with full namespace indication).

        Parameters
        ----------
        namespace: str or None
            If None, all symbols will be returned independently of the namespace.
            If a string, will compare Symbol.namespace to that.
            Defaults to "" which is the namespace for user defined symbols.
        """
        symbols = (s for s in _translators().inspect(self) if isinstance(s, Named))

        namespaces = []
        if namespace is not None:
            namespaces.append(namespace)

        if namespaces:
            symbols = (s for s in symbols if s.namespace in namespaces)

        return set(map(str, symbols))

    def __str__(self):
        return _translators().as_string(self)


@dataclass
class Call(OperandMixin):
    """A Function that has been called with certain arguments."""

    func: Function
    args: tuple
    kwargs_items: tuple[tuple[str, typing.Any], ...] = ()

    def __post_init__(self):
        if isinstance(self.kwargs_items, dict):
            object.__setattr__(self, "kwargs_items", tuple(self.kwargs_items.items()))

    @functools.cached_property
    def kwargs(self):
        return dict(self.kwargs_items)

    def __str__(self):
        return self.func.format(*self.args, *self.kwargs)

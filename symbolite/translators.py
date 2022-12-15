"""
    symbolite.translators
    ~~~~~~~~~~~~~~~~~~~~~

    Translate symbolic expressions to values and strings.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import collections
import types
import typing as ty

from . import lib
from .mappers import (
    AsStr,
    CaptureCount,
    GetItem,
    IdentityMapper,
    Unsupported,
    default_to_name_mapper,
)
from .operands import Call, Symbol, SymbolicExpression

_default_str_mapper = collections.ChainMap(AsStr, default_to_name_mapper)


def map_expression(expr: SymbolicExpression, mapper: GetItem[Symbol, ty.Any]):
    """Map each a symbol recursively.

    Parameters
    ----------
    expr
        symbolic expression.
    mapper
        mapping from symbols to other objects, using getitem.
    """

    if isinstance(expr, Call):
        args = tuple(map_expression(arg, mapper) for arg in expr.args)
        kwargs = {k: map_expression(arg, mapper) for k, arg in expr.kwargs_items}

        f = mapper[expr.func]

        if f is Unsupported:
            raise Unsupported(f"{expr.func} is not supported by this mapper")

        return f(*args, **kwargs)

    if isinstance(expr, Symbol):
        return mapper[expr]

    return expr


def map_expression_by_attr(expr: SymbolicExpression, libsl: types.ModuleType):
    """Map each a symbol recursively.

    Parameters
    ----------
    expr
        symbolic expression.
    libsl
        mapping from symbols to other objects, using getattr.
    """

    if isinstance(expr, Call):
        args = tuple(map_expression_by_attr(arg, libsl) for arg in expr.args)
        kwargs = {k: map_expression_by_attr(arg, libsl) for k, arg in expr.kwargs_items}

        f = getattr(libsl, expr.func.name)

        if f is Unsupported:
            raise Unsupported(f"{expr.func} is not supported by this implementation")

        return f(*args, **kwargs)

    if isinstance(expr, Symbol):
        if expr.namespace == lib.NAMESPACE:
            return getattr(libsl, expr.name)
        if libsl.Symbol is Unsupported:
            raise Unsupported("Symbol is not supported by this implementation")
        return libsl.Symbol(str(expr))

    return expr


def inspect(expr: SymbolicExpression):
    """Inspect an expression and return what is there.
    and within each key there is a dictionary relating the
    given object with the number of times it appears.

    Parameters
    ----------
    expr
        symbolic expression.
    """

    c = CaptureCount()
    map_expression(expr, c)
    return c.content


def replace(expr: SymbolicExpression, *mapers):
    """Replace symbols, functions, values, etc by others.

    If multiple mappers are provided,
        they will be used in order (using a ChainMap)

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    expr
        symbolic expression.
    *mappers
        dictionaries mapping source to destination objects.
    """

    return map_expression(expr, collections.ChainMap(*mapers, IdentityMapper))


def replace_by_name(expr: SymbolicExpression, **symbols):
    """Replace Symbols by values or objects, matching by name.

    If multiple mappers are provided,
        they will be used in order (using a ChainMap)

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    expr
        symbolic expression.
    **symbols
        keyword arguments connecting names to values.
    """

    mapper = {Symbol(k): v for k, v in symbols.items()}
    return map_expression(expr, collections.ChainMap(mapper, IdentityMapper))


def evaluate(
    expr: SymbolicExpression,
    libsl: types.ModuleType,
):
    """Evaluate expression.

    Parameters
    ----------
    expr
        symbolic expression.
    libsl
        implementation module
    """

    return map_expression_by_attr(expr, libsl)


def as_string(
    expr: SymbolicExpression,
    mapper: GetItem[SymbolicExpression, str] = _default_str_mapper,
):
    """Evaluate a call or symbol into a value.

    Parameters
    ----------
    expr
        symbolic expression.
    mapper
        maps user defined symbol. to values.
        defaults to using the same name.
    """
    return map_expression(expr, mapper)


def as_function(
    expr: SymbolicExpression,
    function_name: str,
    params: tuple[str, ...],
    libsl: types.ModuleType,
):
    """Converts the expression to a callable function.

    Parameters
    ----------
    expr
        symbolic expression.
    function_name
        name of the function to be used.
    params
        names of the parameters.s
    libsl:
        implementation module
    """

    function_def = (
        f"""def {function_name}({", ".join(params)}): return {as_string(expr)}"""
    )

    lm = {}
    exec(function_def, dict(libsl=libsl), lm)
    return lm[function_name]

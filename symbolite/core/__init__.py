"""
symbolite.core
~~~~~~~~~~~~~~

Symbolite core classes and functions.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

import collections
import types
import warnings
from functools import singledispatch
from operator import attrgetter
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..impl import find_module_in_stack


class GetImpl(Protocol):
    def get_impl(self, libsl: types.ModuleType) -> Callable[..., Any]: ...


class Unsupported(ValueError):
    """Label unsupported"""


def as_string(expr: Any) -> str:
    """Return the expression as a string.

    Parameters
    ----------
    expr
        symbolic expression.
    """
    return str(expr)


def as_function(
    expr: Any,
    function_name: str,
    params: Sequence[str],
    libsl: types.ModuleType | None = None,
) -> Callable[..., Any]:
    """Converts the expression to a callable function.

    Parameters
    ----------
    expr
        symbolic expression.
    function_name
        name of the function to be used.
    params
        names of the parameters.
    libsl
        implementation module.
    """

    function_def = (
        f"""def {function_name}({", ".join(params)}): return {as_string(expr)}"""
    )

    lm = compile(function_def, libsl)

    return lm[function_name]


def compile(
    code: str,
    libsl: types.ModuleType | None = None,
) -> dict[str, Any]:
    """Compile the code and return the local dictionary.

    Parameters
    ----------
    expr
        symbolic expression.
    libsl
        implementation module.
    """

    if libsl is None:
        libsl = find_module_in_stack()
    if libsl is None:
        warnings.warn("No libsl provided, defaulting to Python standard library.")
        from ..impl import libstd as libsl

    assert libsl is not None

    lm: dict[str, Any] = {}
    exec(
        code,
        {
            "symbol": libsl.symbol,
            "scalar": libsl.scalar,
            "vector": libsl.vector,
            **globals(),
        },
        lm,
    )
    return lm


def inspect(expr: Any) -> dict[Any, int]:
    """Inspect an expression and return what is there
    and how many times.

    Parameters
    ----------
    expr
        symbolic expression.
    """
    if hasattr(expr, "yield_named"):
        cnt = collections.Counter[Any](expr.yield_named())
        return dict(cnt)
    return {expr: 1}


@singledispatch
def evaluate_this(expr: Any, libsl: types.ModuleType) -> Any:
    """Evaluate expression.

    Parameters
    ----------
    expr
        symbolic expression.
    libsl
        implementation module.
    """
    return expr


def evaluate(expr: Any, libsl: types.ModuleType | None = None) -> Any:
    """Evaluate expression.

    Parameters
    ----------
    expr
        symbolic expression.
    libsl
        implementation module.
    """

    if libsl is None:
        libsl = find_module_in_stack()
    if libsl is None:
        warnings.warn("No libsl provided, defaulting to Python standard library.")
        from ..impl import libstd as libsl

    return evaluate_this(expr, libsl)


@singledispatch
def substitute(expr: Any, replacements: Mapping[Any, Any]) -> Any:
    """Replace symbols, functions, values, etc by others.

    Parameters
    ----------
    expr
        symbolic expression.
    replacements
        replacement dictionary.
    """
    return replacements.get(expr, expr)


@singledispatch
def substitute_by_name(expr: Any, **replacements: Any) -> Any:
    """Replace Symbols by values or objects, matching by name.

    Parameters
    ----------
    expr
        symbolic expression.
    replacements
        replacement dictionary.
    """

    if hasattr(expr, "subs_by_name"):
        return expr.subs_by_name(**replacements)
    return expr


def get_impl(expr: Any, libsl: types.ModuleType | None = None) -> Any | Unsupported:
    """Get implementation for a given expression.

    If no implementation library is provided:
    1. 'libsl' will be looked up going back though the stack
        until is found.
    2. If still not found, the implementation using the python
        math module will be used (and a warning will be issued).

    Parameters
    ----------
    expr
        symbolic expression.
    libs
        implementation.
    """
    if libsl is None:
        libsl = find_module_in_stack()
    if hasattr(expr, "get_impl"):
        return expr.get_impl(libsl)
    elif isinstance(expr, str):
        return attrgetter(expr)(libsl)
    else:
        return attrgetter(str(expr))(libsl)

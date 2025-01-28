"""
symbolite.core.operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Common operations to manipulate symbolic expressions.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import collections
import types
import warnings
from functools import singledispatch
from operator import attrgetter
from typing import Any, Callable, Mapping, Sequence

from ..impl import find_module_in_stack


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
    from ..abstract.symbol import yield_named

    cnt = collections.Counter[Any](yield_named(expr))
    if cnt:
        return dict(cnt)
    return {expr: 1}


@singledispatch
def evaluate_impl(expr: Any, libsl: types.ModuleType) -> Any:
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

    return evaluate_impl(expr, libsl)


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
    return expr


@evaluate_impl.register
def evaluate_impl_str(expr: str, libsl: types.ModuleType) -> Any:  # | Unsupported:
    return attrgetter(expr)(libsl)

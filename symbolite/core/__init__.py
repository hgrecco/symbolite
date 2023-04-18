"""
    symbolite.core
    ~~~~~~~~~~~~~~

    Symbolite core classes and functions, includin mappers.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import collections
import types
import typing as ty
import warnings

from ..impl import find_module_in_stack
from . import mappers


def as_string(expr: ty.Any) -> str:
    """Return the expression as a string.

    Parameters
    ----------
    expr
        symbolic expression.
    """

    visitor = mappers.StringifyVisitor()
    return visitor(expr)


def as_function(
    expr: ty.Any,
    function_name: str,
    params: tuple[str, ...],
    libsl: types.ModuleType | None = None,
) -> ty.Callable[..., ty.Any]:
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

    if libsl is None:
        libsl = find_module_in_stack()
    if libsl is None:
        warnings.warn("No libsl provided, defaulting to Python's standard library.")
        from ..impl import libstd as libsl

    lm = {}
    exec(
        function_def,
        {
            "symbol": libsl.symbol,
            "scalar": libsl.scalar,
            "vector": libsl.vector,
            **globals(),
        },
        lm,
    )
    return lm[function_name]


def inspect(expr: ty.Any) -> collections.Counter[ty.Any]:
    """Inspect an expression and return what is there.
    and within each key there is a dictionary relating the
    given object with the number of times it appears.

    Parameters
    ----------
    expr
        symbolic expression.
    """

    visitor = mappers.CounterVisitor()
    visitor(expr)
    return visitor.counter


def evaluate(expr: ty.Any, libsl: types.ModuleType | None = None) -> ty.Any:
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
        warnings.warn("No libsl provided, defaulting to Python's standard library.")
        from ..impl import libstd as libsl

    visitor = mappers.EvaluateVisitor(libsl)
    return visitor(expr)


def substitute(expr: ty.Any, replacements: dict[ty.Any, ty.Any]):
    """Replace symbols, functions, values, etc by others.

    Parameters
    ----------
    expr
        symbolic expression.
    replacements
        replacement dictionary.
    """

    visitor = mappers.SubstituteVisitor(replacements)
    return visitor(expr)


def substitute_by_name(expr: ty.Any, **symbols: ty.Any):
    """Replace Symbols by values or objects, matching by name.

    Parameters
    ----------
    expr
        symbolic expression.
    symbols
        replacement dictionary.
    """

    visitor = mappers.SubstituteByNameVisitor(symbols)
    return visitor(expr)

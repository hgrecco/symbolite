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
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Mapping

from ..impl import find_module_in_stack

if TYPE_CHECKING:
    from ..abstract import Symbol
    from .named import Named


def build_function_code(
    name: str,
    parameters: Iterable[str],
    body: Iterable[str],
    return_variables: Iterable[str],
) -> str:
    """Build function code.

    Parameters
    ----------
    name
        Name of the functions.
    parameters
        Name of the parameters.
    body
        Lines in the body of the function.
    return_variables
        Name of the return variables.
    """

    fdef = (
        f"def {name}({', '.join(parameters)}):\n    "
        + "\n    ".join(body)
        + f"\n    return {', '.join(return_variables)}"
    )
    return fdef


def assign(lhs, rhs):
    return f"{lhs} = {rhs}"


@singledispatch
def as_string(expr: Any) -> str:
    """Return the expression as a string.

    Parameters
    ----------
    expr
        symbolic expression.
    """
    return str(expr)


@singledispatch
def as_function_def(expr) -> str:
    raise TypeError(f"Cannot build function definition for {type(expr)}")


def as_function(
    expr: Any,
    libsl: types.ModuleType | None = None,
) -> Callable[..., Any]:
    """Converts the expression to a callable function.

    Parameters
    ----------
    expr
        symbolic expression.
    libsl
        implementation module.
    """

    function_def = as_function_def(expr)
    lm = compile(function_def, libsl)

    f = lm["f"]
    f.__symbolite_def__ = function_def
    return f


@as_function_def.register(tuple)
@as_function_def.register(list)
def _(
    expr: tuple[Any],
) -> str:
    return build_function_code(
        "f",
        map(str, free_symbols(expr)),
        (assign(f"__return_{ndx}", str(el)) for ndx, el in enumerate(expr)),
        map("__return_{}".format, range(len(expr))),
    )


@as_function_def.register(dict)
def _(
    expr: dict[str, Any],
) -> str:
    return build_function_code(
        "f",
        map(str, free_symbols(tuple(expr.values()))),
        ["__return = {}"] + [assign(f"__return['{k}']", str(el)) for k, el in expr.items()],
        [
            "__return",
        ],
    )


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


@evaluate_impl.register
def evaluate_impl_str(expr: str, libsl: types.ModuleType) -> Any:  # | Unsupported:
    return attrgetter(expr)(libsl)


@singledispatch
def yield_named(
    self: Any, include_anonymous: bool = False
) -> Generator[Named, None, None]:
    return
    yield Named()  # This is required to make it a generator.


@yield_named.register(tuple)
@yield_named.register(list)
def _(expr: tuple[Any]) -> Generator[Named, None, None]:
    for el in expr:
        yield from yield_named(el)


def is_free_symbol(obj: Any) -> bool:
    from ..abstract import Symbol

    return isinstance(obj, Symbol) and obj.expression is None and obj.namespace == ""


def free_symbols(obj: Any) -> tuple[Symbol]:
    seen = []
    for el in filter(is_free_symbol, yield_named(obj)):
        if el not in seen:
            seen.append(el)

    return tuple(seen)

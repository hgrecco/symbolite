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
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping, Sequence

from ..impl import find_module_in_stack

if TYPE_CHECKING:
    from ..abstract import Symbol


def build_function_code(
    name: str,
    parameters: Sequence[str],
    body: Sequence[str],
    return_variables: Sequence[str],
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
    print(fdef)
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
def as_function(
    expr: Any,
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

    function_name = "f"
    function_def = build_function_code(
        function_name,
        tuple(str(s) for s in yield_free_symbols(expr)),
        [
            assign("__out", str(expr)),
        ],
        [
            "__out",
        ],
    )

    lm = compile(function_def, libsl)

    f = lm["f"]
    f.__symbolite_def__ = function_def
    return f


@as_function.register(tuple)
@as_function.register(list)
def _(
    expr: tuple[Any],
    libsl: types.ModuleType | None = None,
) -> Callable[..., Any]:
    function_name = "f"
    function_def = build_function_code(
        "f",
        tuple(str(s) for s in yield_free_symbols(expr)),
        [assign(f"__out_{ndx}", str(el)) for ndx, el in enumerate(expr)],
        [f"__out_{ndx}" for ndx, _ in enumerate(expr)],
    )

    lm = compile(function_def, libsl)

    f = lm[function_name]
    f.__symbolite_def__ = function_def
    return f


@as_function.register(dict)
def _(
    expr: dict[str, Any],
    libsl: types.ModuleType | None = None,
) -> Callable[..., Any]:
    function_name = "f"
    function_def = build_function_code(
        "f",
        tuple(str(s) for s in yield_free_symbols(tuple(expr.values()))),
        ["__out = {}"] + [assign(f"__out['{k}']", str(el)) for k, el in expr.items()],
        [
            "__out",
        ],
    )

    lm = compile(function_def, libsl)

    f = lm[function_name]
    f.__symbolite_def__ = function_def
    return f


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
def yield_free_symbols(expr: Any) -> Generator[Symbol, None, None]:
    return
    yield


@yield_free_symbols.register(tuple)
@yield_free_symbols.register(list)
def _(expr: tuple[Any]) -> Generator[Symbol, Any, types.NoneType]:
    seen = set()
    for el in expr:
        for fs in yield_free_symbols(el):
            if fs in seen:
                continue
            seen.add(fs)
            yield fs

import inspect
import types

import pytest

from symbolite import Symbol, scalar
from symbolite.core import as_function
from symbolite.impl import get_all_implementations

all_impl = get_all_implementations()

x, y, z = map(scalar.Scalar, "x y z".split())

xsy = Symbol("xsy")


@pytest.mark.mypy_testing
def test_typing():
    reveal_type(x + y)  # R: symbolite.abstract.scalar.Scalar # noqa: F821
    reveal_type(2 + y)  # R: symbolite.abstract.scalar.Scalar # noqa: F821
    reveal_type(x + 2)  # R: symbolite.abstract.scalar.Scalar # noqa: F821
    # reveal_type(x + xsy) # R: symbolite.abstract.symbol.Symbol # noqa: F821
    # reveal_type(xsy + x) # R: symbolite.abstract.symbol.Symbol # noqa: F821
    reveal_type(scalar.cos(x))  # R: symbolite.abstract.scalar.Scalar # noqa: F821
    # reveal_type(scalar.cos(xsy)) # R: symbolite.abstract.scalar.Scalar # noqa: F821


@pytest.mark.parametrize(
    "expr",
    [
        x + y,
        x - y,
        x * y,
        x / y,
        x**y,
        x // y,
    ],
)
@pytest.mark.parametrize("libsl", all_impl.values(), ids=all_impl.keys())
def test_known_symbols(expr: Symbol, libsl: types.ModuleType):
    f = as_function(expr, "my_function", ("x", "y"), libsl=libsl)
    assert f.__name__ == "my_function"
    assert expr.subs_by_name(x=2, y=3).eval(libsl=libsl) == f(2, 3)
    assert tuple(inspect.signature(f).parameters.keys()) == ("x", "y")


@pytest.mark.parametrize(
    "expr,replaced",
    [
        (x + scalar.cos(y), 2 + scalar.cos(3)),
        (x + scalar.pi * y, 2 + scalar.pi * 3),
    ],
)
@pytest.mark.parametrize("libsl", all_impl.values(), ids=all_impl.keys())
def test_lib_symbols(expr: Symbol, replaced: Symbol, libsl: types.ModuleType):
    f = as_function(expr, "my_function", ("x", "y"), libsl=libsl)
    value = f(2, 3)
    assert f.__name__ == "my_function"
    assert expr.subs_by_name(x=2, y=3).eval(libsl=libsl) == value
    assert tuple(inspect.signature(f).parameters.keys()) == ("x", "y")


@pytest.mark.parametrize(
    "expr,namespace,result",
    [
        (
            x + scalar.pi * scalar.cos(y),
            None,
            {
                "x",
                "y",
                "scalar.cos",
                "scalar.pi",
                "symbol.mul",
                "symbol.add",
            },
        ),
        (x + scalar.pi * scalar.cos(y), "", {"x", "y"}),
        (
            x + scalar.pi * scalar.cos(y),
            "scalar",
            {"scalar.cos", "scalar.pi"},
        ),
    ],
)
def test_list_symbols(expr: Symbol, namespace: str | None, result: Symbol):
    assert expr.symbol_names(namespace) == result

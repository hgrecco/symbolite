import inspect

import pytest

from symbolite import scalar
from symbolite.core import as_function
from symbolite.impl import get_all_implementations

all_impl = get_all_implementations()

x, y, z = map(scalar.Scalar, "x y z".split())


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
def test_known_symbols(expr, libsl):
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
def test_lib_symbols(expr, replaced, libsl):
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
def test_list_symbols(expr, namespace, result):
    assert expr.symbol_names(namespace) == result

import inspect

import pytest

from symbolite import Symbol, as_function, lib
from symbolite.testsuite.common import all_impl

x, y, z = map(Symbol, "x y z".split())


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
    f = as_function(expr, "my_function", ("x", "y"), libsl)
    assert f.__name__ == "my_function"
    assert expr.replace_by_name(x=2, y=3).eval(libsl) == f(2, 3)
    assert tuple(inspect.signature(f).parameters.keys()) == ("x", "y")


@pytest.mark.parametrize(
    "expr,replaced",
    [
        (x + lib.cos(y), 2 + lib.cos(3)),
        (x + lib.pi * y, 2 + lib.pi * 3),
    ],
)
@pytest.mark.parametrize("libsl", all_impl.values(), ids=all_impl.keys())
def test_lib_symbols(expr, replaced, libsl):
    f = as_function(expr, "my_function", ("x", "y"), libsl)
    value = f(2, 3)
    assert f.__name__ == "my_function"
    assert expr.replace_by_name(x=2, y=3) == replaced
    assert expr.replace_by_name(x=2, y=3).eval(libsl) == value
    assert tuple(inspect.signature(f).parameters.keys()) == ("x", "y")

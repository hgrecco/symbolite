import inspect

import pytest

from symbolite import Symbol, lib
from symbolite.operands import SymbolicExpression
from symbolite.testsuite.common import all_impl
from symbolite.translators import as_function

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


@pytest.mark.parametrize(
    "expr,namespace,skip_operators,result",
    [
        (
            x + lib.pi * lib.cos(y),
            None,
            True,
            {"x", "y", f"{lib.NAMESPACE}.cos", f"{lib.NAMESPACE}.pi"},
        ),
        (
            x + lib.pi * lib.cos(y),
            None,
            False,
            {
                "x",
                "y",
                f"{lib.NAMESPACE}.cos",
                f"{lib.NAMESPACE}.pi",
                f"{lib.NAMESPACE}.op_add",
                f"{lib.NAMESPACE}.op_mul",
            },
        ),
        (x + lib.pi * lib.cos(y), "", True, {"x", "y"}),
        (x + lib.pi * lib.cos(y), "", False, {"x", "y"}),
        (
            x + lib.pi * lib.cos(y),
            "libsl",
            True,
            {f"{lib.NAMESPACE}.cos", f"{lib.NAMESPACE}.pi"},
        ),
        (
            x + lib.pi * lib.cos(y),
            "libsl",
            False,
            {
                f"{lib.NAMESPACE}.cos",
                f"{lib.NAMESPACE}.pi",
                f"{lib.NAMESPACE}.op_add",
                f"{lib.NAMESPACE}.op_mul",
            },
        ),
    ],
)
def test_list_symbols(expr: SymbolicExpression, namespace, skip_operators, result):
    assert expr.symbol_names(namespace, skip_operators) == result

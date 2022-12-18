import inspect

import pytest

from symbolite.abstract import scalar
from symbolite.operands import SymbolicExpression
from symbolite.testsuite.common import all_impl
from symbolite.translators import as_function

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
@pytest.mark.parametrize("libscalar", all_impl.values(), ids=all_impl.keys())
def test_known_symbols(expr, libscalar):
    f = as_function(expr, "my_function", ("x", "y"), libscalar=libscalar)
    assert f.__name__ == "my_function"
    assert expr.replace_by_name(x=2, y=3).eval(libscalar=libscalar) == f(2, 3)
    assert tuple(inspect.signature(f).parameters.keys()) == ("x", "y")


@pytest.mark.parametrize(
    "expr,replaced",
    [
        (x + scalar.cos(y), 2 + scalar.cos(3)),
        (x + scalar.pi * y, 2 + scalar.pi * 3),
    ],
)
@pytest.mark.parametrize("libscalar", all_impl.values(), ids=all_impl.keys())
def test_lib_symbols(expr, replaced, libscalar):
    f = as_function(expr, "my_function", ("x", "y"), libscalar=libscalar)
    value = f(2, 3)
    assert f.__name__ == "my_function"
    assert expr.replace_by_name(x=2, y=3) == replaced
    assert expr.replace_by_name(x=2, y=3).eval(libscalar=libscalar) == value
    assert tuple(inspect.signature(f).parameters.keys()) == ("x", "y")


@pytest.mark.parametrize(
    "expr,namespace,skip_operators,result",
    [
        (
            x + scalar.pi * scalar.cos(y),
            None,
            True,
            {"x", "y", f"{scalar.NAMESPACE}.cos", f"{scalar.NAMESPACE}.pi"},
        ),
        (
            x + scalar.pi * scalar.cos(y),
            None,
            False,
            {
                "x",
                "y",
                f"{scalar.NAMESPACE}.cos",
                f"{scalar.NAMESPACE}.pi",
                f"{scalar.NAMESPACE}.op_add",
                f"{scalar.NAMESPACE}.op_mul",
            },
        ),
        (x + scalar.pi * scalar.cos(y), "", True, {"x", "y"}),
        (x + scalar.pi * scalar.cos(y), "", False, {"x", "y"}),
        (
            x + scalar.pi * scalar.cos(y),
            "libscalar",
            True,
            {f"{scalar.NAMESPACE}.cos", f"{scalar.NAMESPACE}.pi"},
        ),
        (
            x + scalar.pi * scalar.cos(y),
            "libscalar",
            False,
            {
                f"{scalar.NAMESPACE}.cos",
                f"{scalar.NAMESPACE}.pi",
                f"{scalar.NAMESPACE}.op_add",
                f"{scalar.NAMESPACE}.op_mul",
            },
        ),
    ],
)
def test_list_symbols(expr: SymbolicExpression, namespace, skip_operators, result):
    assert expr.symbol_names(namespace, skip_operators) == result

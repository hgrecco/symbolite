import collections

import pytest

from symbolite.abstract import scalar
from symbolite.mappers import AsStr, ToNameMapper
from symbolite.translators import as_string

x, y, z = map(scalar.Scalar, "x y z".split())


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + y, "(x + y)"),
        (x - y, "(x - y)"),
        (x * y, "(x * y)"),
        (x / y, "(x / y)"),
        (x**y, "(x ** y)"),
        (x // y, "(x // y)"),
        (((x**y) % z), "((x ** y) % z)"),
    ],
)
def test_known_symbols(expr, result):
    assert as_string(expr) == result
    assert str(expr) == result


def test_unknown_symbols():
    with pytest.raises(KeyError):
        as_string(x, {})


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + scalar.cos(y), f"(x + {scalar.NAMESPACE}.cos(y))"),
        (x + scalar.pi, f"(x + {scalar.NAMESPACE}.pi)"),
    ],
)
def test_lib_symbols(expr, result):
    mapper = collections.ChainMap(AsStr, ToNameMapper())
    assert as_string(expr, mapper) == result

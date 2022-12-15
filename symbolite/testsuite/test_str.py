import collections

import pytest

from symbolite import Symbol, lib
from symbolite.mappers import AsStr, ToNameMapper
from symbolite.translators import as_string

x, y, z = map(Symbol, "x y z".split())


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
        (x + lib.cos(y), f"(x + {lib.NAMESPACE}.cos(y))"),
        (x + lib.pi, f"(x + {lib.NAMESPACE}.pi)"),
    ],
)
def test_lib_symbols(expr, result):
    mapper = collections.ChainMap(AsStr, ToNameMapper())
    assert as_string(expr, mapper) == result

import collections

import pytest

from symbolite import Symbol, as_string, lib
from symbolite.mappers import AsStr, ToNameMapper
from symbolite.operands import NAMESPACE

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
        (x + lib.cos(y), f"(x + {NAMESPACE}.cos(y))"),
        (x + lib.pi, f"(x + {NAMESPACE}.pi)"),
    ],
)
def test_lib_symbols(expr, result):
    mapper = collections.ChainMap(AsStr, ToNameMapper())
    assert as_string(expr, mapper) == result


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + lib.cos(y) * lib.pi, f"(m_x + ({NAMESPACE}.cos(m_y) * {NAMESPACE}.pi))"),
    ],
)
def test_known_symbols_prefixed(expr, result):
    mapper = collections.ChainMap(AsStr, ToNameMapper("m_"))
    assert as_string(expr, mapper) == result

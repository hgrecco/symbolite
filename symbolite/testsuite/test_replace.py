import pytest

from symbolite import Symbol, lib
from symbolite.translators import replace, replace_by_name

x, y, z = map(Symbol, "x y z".split())


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + 2 * y, x + 2 * z),
        (x + 2 * lib.cos(y), x + 2 * lib.cos(z)),
    ],
)
def test_replace(expr, result):
    assert replace(expr, {Symbol("y"): Symbol("z")}) == result
    assert expr.replace({Symbol("y"): Symbol("z")}) == result


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + 2 * y, x + 2 * z),
        (x + 2 * lib.cos(y), x + 2 * lib.cos(z)),
    ],
)
def test_replace_by_name(expr, result):
    assert replace_by_name(expr, y=Symbol("z")) == result
    assert expr.replace_by_name(y=Symbol("z")) == result

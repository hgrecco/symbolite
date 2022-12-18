import pytest

from symbolite import Scalar
from symbolite.abstract import scalar
from symbolite.translators import replace, replace_by_name

x, y, z = map(Scalar, "x y z".split())


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + 2 * y, x + 2 * z),
        (x + 2 * scalar.cos(y), x + 2 * scalar.cos(z)),
    ],
)
def test_replace(expr, result):
    assert replace(expr, {Scalar("y"): Scalar("z")}) == result
    assert expr.replace({Scalar("y"): Scalar("z")}) == result


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + 2 * y, x + 2 * z),
        (x + 2 * scalar.cos(y), x + 2 * scalar.cos(z)),
    ],
)
def test_replace_by_name(expr, result):
    assert replace_by_name(expr, y=Scalar("z")) == result
    assert expr.replace_by_name(y=Scalar("z")) == result

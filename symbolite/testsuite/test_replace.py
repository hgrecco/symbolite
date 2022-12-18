import pytest

from symbolite.abstract import scalar
from symbolite.translators import replace, replace_by_name

x, y, z = map(scalar.Scalar, "x y z".split())


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + 2 * y, x + 2 * z),
        (x + 2 * scalar.cos(y), x + 2 * scalar.cos(z)),
    ],
)
def test_replace(expr, result):
    assert replace(expr, {scalar.Scalar("y"): scalar.Scalar("z")}) == result
    assert expr.replace({scalar.Scalar("y"): scalar.Scalar("z")}) == result


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + 2 * y, x + 2 * z),
        (x + 2 * scalar.cos(y), x + 2 * scalar.cos(z)),
    ],
)
def test_replace_by_name(expr, result):
    assert replace_by_name(expr, y=scalar.Scalar("z")) == result
    assert expr.replace_by_name(y=scalar.Scalar("z")) == result

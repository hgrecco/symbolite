import math as pymath

import pytest

from symbolite import Symbol, lib
from symbolite.testsuite.common import all_impl
from symbolite.translators import evaluate

x, y, z = map(Symbol, "x y z".split())


@pytest.mark.parametrize(
    "expr,result",
    [
        (2 * lib.cos(0.5), 2 * pymath.cos(0.5)),
    ],
)
def test_evaluate(expr, result):
    assert evaluate(expr, all_impl["math"]) == result
    assert expr.eval(all_impl["math"]) == result

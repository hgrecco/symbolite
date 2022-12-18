import math as pymath

import pytest

from symbolite.abstract import scalar
from symbolite.testsuite.common import all_impl
from symbolite.translators import evaluate

x, y, z = map(scalar.Scalar, "x y z".split())


@pytest.mark.parametrize(
    "expr,result",
    [
        (2 * scalar.cos(0.5), 2 * pymath.cos(0.5)),
    ],
)
def test_evaluate(expr, result):
    assert evaluate(expr, libscalar=all_impl["default"]) == result
    assert expr.eval(libscalar=all_impl["default"]) == result

import pytest

from symbolite import scalar, vector
from symbolite.core.base import Unsupported
from symbolite.impl import get_all_implementations

all_impl = get_all_implementations()

x, y = map(scalar.Scalar, ("x", "y"))
vec = vector.Vector("vec")
v = vector.Vector("v")


def test_vector():
    assert str(vec) == "vec"
    assert str(vec[1]) == "vec[1]"


def test_methods():
    assert vec.subs_by_name(vec=(1, 2, 3)) == (1, 2, 3)
    assert vec[1].subs_by_name(vec=(1, 2, 3)).eval() == 2
    assert vec.symbol_names() == {
        "vec",
    }
    assert vec[1].symbol_names() == {
        "vec",
    }
    assert (vec[1] + vec[0]).symbol_names() == {
        "vec",
    }


@pytest.mark.parametrize("libsl", all_impl.values(), ids=all_impl.keys())
def test_impl(libsl):
    v = (1, 2, 3, 4)

    try:
        expr = vector.sum(v)
        assert expr.eval(libsl=libsl) == 10
    except Unsupported:
        pass

    expr = vector.prod(v)
    assert expr.eval(libsl=libsl) == 24


def test_impl_numpy():
    try:
        import numpy as np

        from symbolite.impl.libnumpy import libnumpy as libsl
    except ImportError:
        return

    v = np.asarray((1, 2, 3))

    expr = vector.Vector("vec") + 1
    assert np.allclose(expr.replace_by_name(vec=v).eval(), v + 1)

    expr = scalar.cos(vector.Vector("vec"))

    assert np.allclose(expr.substitute_by_name(vec=v).eval(libsl=libsl), np.cos(v))


def test_impl_sympy():
    try:
        import sympy as sy

        from symbolite.impl import libsympy as libsl
    except ImportError:
        return

    vec = vector.Vector("vec")
    syarr = sy.IndexedBase("vec")
    assert vec.eval(libsl=libsl) == syarr
    assert vec[1].eval(libsl=libsl) == syarr[1]


@pytest.mark.parametrize(
    "expr,params,result",
    [
        (x + 2 * y, ("x", "y"), vec[0] + 2 * vec[1]),
        (x + 2 * y, ("y", "x"), vec[1] + 2 * vec[0]),
        (x + 2 * scalar.cos(y), ("y", "x"), vec[1] + 2 * scalar.cos(vec[0])),
        (x + 2 * y, dict(x=3, y=5), vec[3] + 2 * vec[5]),
        (x + 2 * y, dict(x=5, y=3), vec[5] + 2 * vec[3]),
    ],
)
def test_vectorize(expr, params, result):
    assert vector.vectorize(expr, params, result)


def test_vectorize_non_default_varname():
    assert vector.vectorize(x + 2 * y, ("x", "y"), v[0] + 2 * v[1])


@pytest.mark.parametrize(
    "expr,result",
    [
        (x + 2 * y, (("x", "y"), vec[0] + 2 * vec[1])),
        (y + 2 * x, (("x", "y"), vec[1] + 2 * vec[0])),
        (x + 2 * scalar.cos(y), (("x", "y"), vec[0] + 2 * scalar.cos(vec[1]))),
    ],
)
def test_autovectorize(expr, result):
    assert vector.auto_vectorize(expr) == result


def test_autovectorize_non_default_varname():
    assert vector.auto_vectorize(x + 2 * y, "v") == (("x", "y"), v[0] + 2 * v[1])

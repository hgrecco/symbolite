import math

from symbolite import Scalar, Symbol, scalar
from symbolite.core.named import symbol_names
from symbolite.core.operations import as_string, evaluate, substitute
from symbolite.core.symbolgroup import AutoSymbol, SymbolicNamespace
from symbolite.impl import libstd


def test_naming():
    class N(SymbolicNamespace):
        x = AutoSymbol()
        y = AutoSymbol()

    assert isinstance(N.x, Symbol)
    assert isinstance(N.y, Symbol)
    assert N.x.name == "x"
    assert N.y.name == "y"
    assert symbol_names(N) == {"x", "y"}


def test_substitute_eval():
    class X(SymbolicNamespace):
        x = Scalar("x")
        y = Scalar("y")
        z = x + 2 * y
        p = scalar.cos(z)

    Y = substitute(X, {X.x: 1, X.y: 2})

    assert Y.x == 1
    assert Y.y == 2

    d = evaluate(Y, libstd)
    assert d["x"] == 1
    assert d["y"] == 2
    assert d["z"] == d["x"] + 2 * d["y"]
    assert d["p"] == math.cos(d["z"])


def test_as_str():
    class X(SymbolicNamespace):
        x = Scalar("x")
        y = Scalar("y")
        z = x + 2 * y
        p = scalar.cos(z)

    s = "def X(x, y):\n    p = scalar.cos(x + 2 * y)\n    z = x + 2 * y"

    assert as_string(X) == s


def test_eq():
    class N(SymbolicNamespace):
        x = AutoSymbol()
        y = AutoSymbol()

        x.eq(2 * y)

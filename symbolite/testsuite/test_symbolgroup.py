from symbolite import Symbol
from symbolite.core.symbolgroup import AutoSymbol, SymbolicNamespace


def test_naming():
    class N(SymbolicNamespace):
        x = AutoSymbol()
        y = AutoSymbol()

    assert isinstance(N.x, Symbol)
    assert isinstance(N.y, Symbol)
    assert N.x.name == "x"
    assert N.y.name == "y"
    assert N.symbol_names() == {"x", "y"}


def test_eq():
    class N(SymbolicNamespace):
        x = AutoSymbol()
        y = AutoSymbol()

        x.eq(2 * y)

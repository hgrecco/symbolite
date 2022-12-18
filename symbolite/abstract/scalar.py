"""
    symbolite.lib
    ~~~~~~~~~~~~~

    Function and values known by symbolite.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import functools
import operator

from symbolite.operands import Function, Named, Operator, SymbolicExpression

NAMESPACE = "libscalar"


_functions = (
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "comb",
    "copysign",
    "cos",
    "cosh",
    "degrees",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "fabs",
    "factorial",
    "floor",
    "fmod",
    "frexp",
    "gamma",
    "gcd",
    "hypot",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isqrt",
    "lcm",
    "ldexp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "modf",
    "nextafter",
    "perm",
    "pow",
    "radians",
    "remainder",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    "ulp",
)

_values = ("e", "inf", "pi", "nan", "tau")

_operators = {
    "op_modpow": Operator.from_operator(pow, "(({} ** {}) % {})", NAMESPACE),
    "op_add": Operator.from_operator(operator.add, "({} + {})", NAMESPACE),
    "op_sub": Operator.from_operator(operator.sub, "({} - {})", NAMESPACE),
    "op_mul": Operator.from_operator(operator.mul, "({} * {})", NAMESPACE),
    "op_truediv": Operator.from_operator(operator.truediv, "({} / {})", NAMESPACE),
    "op_floordiv": Operator.from_operator(operator.floordiv, "({} // {})", NAMESPACE),
    "op_pow": Operator.from_operator(operator.pow, "({} ** {})", NAMESPACE),
    "op_mod": Operator.from_operator(operator.mod, "({} % {})", NAMESPACE),
    "op_pos": Operator.from_operator(operator.pos, "(+{})", NAMESPACE),
    "op_neg": Operator.from_operator(operator.neg, "(-{})", NAMESPACE),
}


@dataclasses.dataclass(frozen=True)
class Scalar(Named, SymbolicExpression):
    """A user defined symbol."""


__all__ = sorted(_values + _functions + tuple(_operators.keys()) + ("Scalar",))


def __dir__():
    return __all__


@functools.lru_cache(maxsize=None)
def __getattr__(name):

    if name not in __all__:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _operators:
        return _operators[name]
    elif name in _values:
        return Scalar(name, NAMESPACE)
    else:
        return Function(name, NAMESPACE)

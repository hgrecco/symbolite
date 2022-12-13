"""
    symbolite.lib
    ~~~~~~~~~~~~~

    Function and values known by symbolite.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import functools
import operator

from .operands import LIBPREFIX, Function, Operator, Symbol

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
    "dist",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "fabs",
    "factorial",
    "floor",
    "fmod",
    "frexp",
    "fsum",
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
    "prod",
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
    "op_modpow": Operator.from_operator(pow, "(({} ** {}) % {})"),
    "op_add": Operator.from_operator(operator.add, "({} + {})"),
    "op_sub": Operator.from_operator(operator.sub, "({} - {})"),
    "op_mul": Operator.from_operator(operator.mul, "({} * {})"),
    "op_truediv": Operator.from_operator(operator.truediv, "({} / {})"),
    "op_floordiv": Operator.from_operator(operator.floordiv, "({} // {})"),
    "op_pow": Operator.from_operator(operator.pow, "({} ** {})"),
    "op_mod": Operator.from_operator(operator.mod, "({} % {})"),
    "op_pos": Operator.from_operator(operator.pos, "(+{})"),
    "op_neg": Operator.from_operator(operator.neg, "(-{})"),
}

__all__ = sorted(_values + _functions + tuple(_operators.keys()))


def __dir__():
    return __all__


@functools.lru_cache(maxsize=None)
def __getattr__(name):

    if name not in __all__:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _operators:
        return _operators[name]
    elif name in _values:
        return Symbol(name, LIBPREFIX)
    else:
        return Function(name, LIBPREFIX)

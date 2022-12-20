"""
    symbolite.abstract.scalar
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Function and values for scalar operations.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import functools
import operator

from symbolite.operands import Function, Named, Operator, SymbolicExpression

NAMESPACE = "libscalar"


_functions = {
    "abs": 1,
    "acos": 1,
    "acosh": 1,
    "asin": 1,
    "asinh": 1,
    "atan": 1,
    "atan2": 2,
    "atanh": 1,
    "ceil": 1,
    "comb": 2,
    "copysign": 2,
    "cos": 1,
    "cosh": 1,
    "degrees": 1,
    "erf": 1,
    "erfc": 1,
    "exp": 1,
    "expm1": 1,
    "fabs": 1,
    "factorial": 1,
    "floor": 1,
    "fmod": 2,
    "frexp": 1,
    "gamma": 1,
    "gcd": None,  # 1 to ---
    "hypot": None,  # 1 to ---
    "isclose": None,  # 2, 3, 4
    "isfinite": 1,
    "isinf": 1,
    "isnan": 1,
    "isqrt": 1,
    "lcm": None,  # 1 to ---
    "ldexp": 2,
    "lgamma": 1,
    "log": None,  # 1 or 2
    "log10": 1,
    "log1p": 1,
    "log2": 1,
    "modf": 1,
    "nextafter": 2,
    "perm": None,  # 1 or 2
    "pow": 2,
    "radians": 1,
    "remainder": 2,
    "sin": 1,
    "sinh": 1,
    "sqrt": 1,
    "tan": 1,
    "tanh": 1,
    "trunc": 1,
    "ulp": 1,
}

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


__all__ = sorted(
    _values + tuple(_functions.keys()) + tuple(_operators.keys()) + ("Scalar",)
)


def __dir__():
    return __all__


@functools.lru_cache(maxsize=None)
def __getattr__(name):

    if name not in __all__:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _operators:
        return _operators[name]
    elif name in _functions:
        return Function(name, NAMESPACE, arity=_functions[name])
    else:
        return Scalar(name, NAMESPACE)

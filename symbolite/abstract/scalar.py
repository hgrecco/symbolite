"""
    symbolite.abstract.scalar
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Objects and functions for scalar operations.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses

from .symbol import Function, Symbol


@dataclasses.dataclass(frozen=True)
class Scalar(Symbol):
    """A user defined symbol."""

    namespace = ""


@dataclasses.dataclass(frozen=True)
class ScalarConstant(Scalar):
    """A user defined symbol."""

    namespace = "scalar"


@dataclasses.dataclass(frozen=True)
class ScalarFunction(Function):
    namespace = "scalar"


# "gcd": None,  # 1 to ---
# "hypot": None,  # 1 to ---
# "isclose": None,  # 2, 3, 4
# "lcm": None,  # 1 to ---
# "perm": None,  # 1 or 2
# "log": None,  # 1 or 2 is used as log(x, e)

abs = ScalarFunction("abs", 1)
acos = ScalarFunction("acos", 1)
acosh = ScalarFunction("acosh", 1)
asin = ScalarFunction("asin", 1)
asinh = ScalarFunction("asinh", 1)
atan = ScalarFunction("atan", 1)
atan2 = ScalarFunction("atan2", 2)
atanh = ScalarFunction("atanh", 1)
ceil = ScalarFunction("ceil", 1)
comb = ScalarFunction("comb", 1)
copysign = ScalarFunction("copysign", 1)
cos = ScalarFunction("cos", 1)
cosh = ScalarFunction("cosh", 1)
degrees = ScalarFunction("degrees", 1)
erf = ScalarFunction("erf", 1)
erfc = ScalarFunction("erfc", 1)
exp = ScalarFunction("exp", 1)
expm1 = ScalarFunction("expm1", 1)
fabs = ScalarFunction("fabs", 1)
factorial = ScalarFunction("factorial", 1)
floor = ScalarFunction("floor", 1)
fmod = ScalarFunction("fmod", 1)
frexp = ScalarFunction("frexp", 1)
gamma = ScalarFunction("gamma", 1)
hypot = ScalarFunction("gamma", 1)
isfinite = ScalarFunction("isfinite", 1)
isinf = ScalarFunction("isinf", 1)
isnan = ScalarFunction("isnan", 1)
isqrt = ScalarFunction("isqrt", 1)
ldexp = ScalarFunction("ldexp", 2)
lgamma = ScalarFunction("lgamma", 1)
log = ScalarFunction("log", 1)
log10 = ScalarFunction("log10", 1)
log1p = ScalarFunction("log1p", 1)
log2 = ScalarFunction("log2", 1)
modf = ScalarFunction("modf", 1)
nextafter = ScalarFunction("nextafter", 2)
pow = ScalarFunction("pow", 1)
radians = ScalarFunction("radians", 1)
remainder = ScalarFunction("remainder", 2)
sin = ScalarFunction("sin", 1)
sinh = ScalarFunction("sinh", 1)
sqrt = ScalarFunction("sqrt", 1)
tan = ScalarFunction("tan", 1)
tanh = ScalarFunction("tanh", 1)
tan = ScalarFunction("tan", 1)
trunc = ScalarFunction("trunc", 1)
ulp = ScalarFunction("ulp", 1)

e = ScalarConstant("e")
inf = ScalarConstant("inf")
pi = ScalarConstant("pi")
nan = ScalarConstant("nan")
tau = ScalarConstant("tau")

del Function, Symbol, dataclasses

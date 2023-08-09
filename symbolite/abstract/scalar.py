"""
    symbolite.abstract.scalar
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Objects and functions for scalar operations.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..core import Unsupported
from .symbol import BaseFunction, Symbol

NumberT = int | float | complex


@dataclasses.dataclass(frozen=True)
class Scalar(Symbol):
    """A user defined symbol."""

    def __getitem__(self, key: Any):
        return Unsupported

    def __getattr__(self, key: Any):
        raise AttributeError(key)


@dataclasses.dataclass(frozen=True)
class ScalarUnaryFunction(BaseFunction):
    namespace: str = "scalar"
    arity: int = 1

    def __call__(self, arg1: Scalar | NumberT) -> Scalar:
        return super()._call(arg1)  # type: ignore


@dataclasses.dataclass(frozen=True)
class ScalarBinaryFunction(BaseFunction):
    namespace: str = "scalar"
    arity: int = 2

    def __call__(self, arg1: Scalar | NumberT, arg2: Scalar | NumberT) -> Scalar:
        return super()._call(arg1, arg2)  # type: ignore


# "gcd": None,  # 1 to ---
# "hypot": None,  # 1 to ---
# "isclose": None,  # 2, 3, 4
# "lcm": None,  # 1 to ---
# "perm": None,  # 1 or 2
# "log": None,  # 1 or 2 is used as log(x, e)

abs = ScalarUnaryFunction("abs")
acos = ScalarUnaryFunction("acos")
acosh = ScalarUnaryFunction("acosh")
asin = ScalarUnaryFunction("asin")
asinh = ScalarUnaryFunction("asinh")
atan = ScalarUnaryFunction("atan")
atan2 = ScalarBinaryFunction("atan2")
atanh = ScalarUnaryFunction("atanh")
ceil = ScalarUnaryFunction("ceil")
comb = ScalarUnaryFunction("comb")
copysign = ScalarUnaryFunction("copysign")
cos = ScalarUnaryFunction("cos")
cosh = ScalarUnaryFunction("cosh")
degrees = ScalarUnaryFunction("degrees")
erf = ScalarUnaryFunction("erf")
erfc = ScalarUnaryFunction("erfc")
exp = ScalarUnaryFunction("exp")
expm1 = ScalarUnaryFunction("expm1")
fabs = ScalarUnaryFunction("fabs")
factorial = ScalarUnaryFunction("factorial")
floor = ScalarUnaryFunction("floor")
fmod = ScalarUnaryFunction("fmod")
frexp = ScalarUnaryFunction("frexp")
gamma = ScalarUnaryFunction("gamma")
hypot = ScalarUnaryFunction("gamma")
isfinite = ScalarUnaryFunction("isfinite")
isinf = ScalarUnaryFunction("isinf")
isnan = ScalarUnaryFunction("isnan")
isqrt = ScalarUnaryFunction("isqrt")
ldexp = ScalarBinaryFunction("ldexp")
lgamma = ScalarUnaryFunction("lgamma")
log = ScalarUnaryFunction("log")
log10 = ScalarUnaryFunction("log10")
log1p = ScalarUnaryFunction("log1p")
log2 = ScalarUnaryFunction("log2")
modf = ScalarUnaryFunction("modf")
nextafter = ScalarUnaryFunction("nextafter")
pow = ScalarUnaryFunction("pow")
radians = ScalarUnaryFunction("radians")
remainder = ScalarBinaryFunction("remainder")
sin = ScalarUnaryFunction("sin")
sinh = ScalarUnaryFunction("sinh")
sqrt = ScalarUnaryFunction("sqrt")
tan = ScalarUnaryFunction("tan")
tanh = ScalarUnaryFunction("tanh")
tan = ScalarUnaryFunction("tan")
trunc = ScalarUnaryFunction("trunc")
ulp = ScalarUnaryFunction("ulp")

e = Scalar("e", namespace="scalar")
inf = Scalar("inf", namespace="scalar")
pi = Scalar("pi", namespace="scalar")
nan = Scalar("nan", namespace="scalar")
tau = Scalar("tau", namespace="scalar")

del BaseFunction, Symbol, dataclasses

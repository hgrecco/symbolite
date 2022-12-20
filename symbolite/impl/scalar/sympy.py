"""
    symbolite.impl.scalar.sympy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Translate symbolite.lib into values and functions
    defined in SymPy.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import operator

import sympy as sy
from sympy.abc import x, y

from symbolite.mappers import Unsupported

abs = sy.Abs
acos = sy.acos
acosh = sy.acosh
asin = sy.asin
asinh = sy.asinh
atan = sy.atan
atan2 = sy.atan2
atanh = sy.atanh
ceil = sy.ceiling
comb = Unsupported
copysign = Unsupported
cos = sy.cos
cosh = sy.cosh
degrees = sy.Lambda(x, x * 180 / sy.pi)
erf = Unsupported
erfc = Unsupported
exp = sy.exp
expm1 = sy.Lambda(x, sy.exp(x) - 1)
fabs = sy.Abs
factorial = Unsupported
floor = sy.floor
fmod = sy.Mod
frexp = Unsupported
gamma = Unsupported
gcd = Unsupported
hypot = sy.Lambda((x, y), sy.sqrt(x * x + y * y))
isclose = Unsupported
isfinite = Unsupported
isinf = Unsupported
isnan = Unsupported
isqrt = Unsupported
lcm = sy.lcm
ldexp = Unsupported
lgamma = sy.loggamma
log = sy.log
log10 = sy.Lambda(x, sy.log(x, 10))
log1p = sy.Lambda(x, sy.log(1 + x))
log2 = sy.Lambda(x, sy.log(x, 2))
modf = Unsupported
nextafter = Unsupported
perm = Unsupported
pow = sy.Pow
radians = sy.Lambda(x, x * sy.pi / 180)
remainder = Unsupported
sin = sy.sin
sinh = sy.sinh
sqrt = sy.sqrt
tan = sy.tan
tanh = sy.tanh
trunc = Unsupported
ulp = Unsupported

e = sy.exp(1)
inf = sy.oo
pi = sy.pi
nan = sy.nan
tau = 2 * sy.pi

op_modpow = pow
op_add = operator.add
op_sub = operator.sub
op_mul = operator.mul
op_truediv = operator.truediv
op_floordiv = operator.floordiv
op_pow = operator.pow
op_mod = operator.mod
op_pos = operator.pos
op_neg = operator.neg

Scalar = sy.Symbol

del sy, operator, Unsupported, x, y

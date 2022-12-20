"""
    symbolite.impl.scalar.default
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Translate symbolite.lib into values and functions
    defined in Python's math module.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import math
import operator

from symbolite.mappers import Unsupported

_abs = abs
_pow = pow

abs = _abs
acos = math.acos
acosh = math.acosh
asin = math.asin
asinh = math.asinh
atan = math.atan
atan2 = math.atan2
atanh = math.atanh
ceil = math.ceil
comb = math.comb
copysign = math.copysign
cos = math.cos
cosh = math.cosh
degrees = math.degrees
erf = math.erf
erfc = math.erfc
exp = math.exp
expm1 = math.expm1
fabs = math.fabs
factorial = math.factorial
floor = math.floor
fmod = math.fmod
frexp = math.frexp
gamma = math.gamma
gcd = math.gcd
hypot = math.hypot
isclose = math.isclose
isfinite = math.isfinite
isinf = math.isinf
isnan = math.isnan
isqrt = math.isqrt
lcm = getattr(math, "lcm", Unsupported)  # New in Python Version 3.9
ldexp = math.ldexp
lgamma = math.lgamma
log = math.log
log10 = math.log10
log1p = math.log1p
log2 = math.log2
modf = math.modf
nextafter = getattr(math, "nextafter", Unsupported)  # New in Python Version 3.9
perm = math.perm
pow = math.pow
radians = math.radians
remainder = math.remainder
sin = math.sin
sinh = math.sinh
sqrt = math.sqrt
tan = math.tan
tanh = math.tanh
trunc = math.trunc
ulp = getattr(math, "ulp", Unsupported)  # New in Python Version 3.9

e = math.e
inf = math.inf
pi = math.pi
nan = math.nan
tau = math.tau

op_modpow = _pow
op_add = operator.add
op_sub = operator.sub
op_mul = operator.mul
op_truediv = operator.truediv
op_floordiv = operator.floordiv
op_pow = operator.pow
op_mod = operator.mod
op_pos = operator.pos
op_neg = operator.neg

Scalar = Unsupported

del math, operator, Unsupported

"""
    symbolite.impl.scalar.numpy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Translate symbolite.lib into values and functions
    defined in NumPy.

    :copyright: 2022 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import operator

import numpy as np

from symbolite.mappers import Unsupported

abs = np.abs
acos = np.arccos
acosh = np.arccosh
asin = np.arcsin
asinh = np.arcsinh
atan = np.arctan
atan2 = np.arctan2
atanh = np.arctanh
ceil = np.ceil
comb = Unsupported
copysign = np.copysign
cos = np.cos
cosh = np.cosh
degrees = np.degrees
erf = Unsupported
erfc = Unsupported
exp = np.exp
expm1 = np.expm1
fabs = np.fabs
factorial = Unsupported
floor = np.floor
fmod = np.fmod
frexp = np.frexp
gamma = Unsupported
gcd = Unsupported
hypot = np.hypot
isclose = np.isclose
isfinite = np.isfinite
isinf = np.isinf
isnan = np.isnan
isqrt = Unsupported
lcm = np.lcm
ldexp = np.ldexp
lgamma = Unsupported
log = np.log
log10 = np.log10
log1p = np.log1p
log2 = np.log2
modf = np.modf
nextafter = np.nextafter
perm = Unsupported
pow = np.power
radians = np.radians
remainder = np.remainder
sin = np.sin
sinh = np.sinh
sqrt = np.sqrt
tan = np.tan
tanh = np.tanh
trunc = np.trunc
ulp = Unsupported

e = np.e
inf = np.inf
pi = np.pi
nan = np.nan
tau = 2 * np.pi

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

Scalar = Unsupported

del np, operator, Unsupported

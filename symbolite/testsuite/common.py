from symbolite.impl.scalar import default

all_impl = {"default": default}

try:
    from symbolite.impl.scalar import numpy as npm

    all_impl["numpy"] = npm
except ImportError:
    pass

try:
    from symbolite.impl.scalar import sympy as spm

    all_impl["sympy"] = spm
except ImportError:
    pass

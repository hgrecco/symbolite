from symbolite.libimpl import math as mpm

all_impl = {"math": mpm}

try:
    from symbolite.libimpl import numpy as npm

    all_impl["numpy"] = npm
except ImportError:
    pass

try:
    from symbolite.libimpl import sympy as spm

    all_impl["sympy"] = spm
except ImportError:
    pass

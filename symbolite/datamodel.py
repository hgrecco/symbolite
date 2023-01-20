"""
    symbolite.datamodel
    ~~~~~~~~~~~~~~~~~~~

    Functions returned by an instance magic methods.
    Normally this will defined within the abstract symbol module,
    but we do this here as not only symbols, but also calls require to
    "know" how to deal with these operations.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from .operands import Function

# Follow the order in
# https://docs.python.org/3/reference/datamodel.html

# comparison.
__lt__ = Function("__lt__", "libsymbol", 2, "({} < {})")
__le__ = Function("__le__", "libsymbol", 2, "({} <= {})")
__eq__ = Function("__eq__", "libsymbol", 2, "({} == {})")
__ne__ = Function("__ne__", "libsymbol", 2, "({} != {})")
__gt__ = Function("__gt__", "libsymbol", 2, "({} > {})")
__ge__ = Function("__ge__", "libsymbol", 2, "({} >= {})")

# Emulating container types
__getitem__ = Function("__getitem__", "libsymbol", 2, "{}[{}]")

# Emulating numeric types
__add__ = Function("__add__", "libsymbol", 2, "({} + {})")
__sub__ = Function("__sub__", "libsymbol", 2, "({} - {})")
__mul__ = Function("__mul__", "libsymbol", 2, "({} * {})")
__matmul__ = Function("__matmul__", "libsymbol", 2, "({} @ {})")
__truediv__ = Function("__truediv__", "libsymbol", 2, "({} / {})")
__floordiv__ = Function("__floordiv__", "libsymbol", 2, "({} // {})")
__mod__ = Function("__mod__", "libsymbol", 2, "({} % {})")
__pow__ = Function("__pow__", "libsymbol", 2, "({} ** {})")
__pow3__ = Function("__pow3__", "libsymbol", 3, "pow({}, {}, {})")
__lshift__ = Function("__lshift__", "libsymbol", 2, "({} << {})")
__rshift__ = Function("__rshift__", "libsymbol", 2, "({} >> {})")
__and__ = Function("__and__", "libsymbol", 2, "({} & {})")
__xor__ = Function("__xor__", "libsymbol", 2, "({} ^ {})")
__or__ = Function("__or__", "libsymbol", 2, "({} | {})")

# reflective versions
__radd__ = Function("__radd__", "libsymbol", 2, "({1} + {0})")
__rsub__ = Function("__rsub__", "libsymbol", 2, "({1} - {0})")
__rmul__ = Function("__rmul__", "libsymbol", 2, "({1} * {0})")
__rmatmul__ = Function("__rmatmul__", "libsymbol", 2, "({1} @ {0})")
__rtruediv__ = Function("__rtruediv__", "libsymbol", 2, "({1} / {0})")
__rfloordiv__ = Function("__rfloordiv__", "libsymbol", 2, "({1} // {0})")
__rmod__ = Function("__rmod__", "libsymbol", 2, "({1} % {0})")
__rpow__ = Function("__pow__", "libsymbol", 2, "({1} ** {0})")
__rlshift__ = Function("__rlshift__", "libsymbol", 2, "({1} << {0})")
__rrshift__ = Function("__rrshift__", "libsymbol", 2, "({1} >> {0})")
__rand__ = Function("__rand__", "libsymbol", 2, "({1} & {0})")
__rxor__ = Function("__rxor__", "libsymbol", 2, "({1} ^ {0})")
__ror__ = Function("__ror__", "libsymbol", 2, "({1} | {0})")

__neg__ = Function("__neg__", "libsymbol", 1, "(-{})")
__pos__ = Function("__pos__", "libsymbol", 1, "(+{})")
__invert__ = Function("__invert__", "libsymbol", 1, "(~{})")

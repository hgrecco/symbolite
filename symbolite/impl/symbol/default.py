"""
    symbolite.libsymbolite
    ~~~~~~~~~~~~~~~~~~~~~~

    Operators related to magic methods.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import operator

__lt__ = operator.lt
__le__ = operator.le
__eq__ = operator.eq
__ne__ = operator.ne
__gt__ = operator.gt
__ge__ = operator.ge

__getitem__ = operator.getitem

__add__ = operator.add
__sub__ = operator.sub
__mul__ = operator.mul
__matmul__ = operator.matmul
__truediv__ = operator.truediv
__floordiv__ = operator.floordiv
__mod__ = operator.mod
__pow__ = operator.pow
__pow3__ = pow
__lshift__ = operator.lshift
__rshift__ = operator.rshift
__and__ = operator.and_
__xor__ = operator.xor
__or__ = operator.or_


def _rev(func):
    return lambda a, b: func(b, a)


__radd__ = _rev(operator.add)
__rsub__ = _rev(operator.sub)
__rmul__ = _rev(operator.mul)
__rmatmul__ = _rev(operator.matmul)
__rtruediv__ = _rev(operator.truediv)
__rfloordiv__ = _rev(operator.floordiv)
__rmod__ = _rev(operator.mod)
__rpow__ = _rev(operator.pow)
__rlshift__ = _rev(operator.lshift)
__rrshift__ = _rev(operator.rshift)
__rand__ = _rev(operator.and_)
__rxor__ = _rev(operator.xor)
__ror__ = _rev(operator.or_)

__neg__ = operator.neg
__pos__ = operator.pos
__invert__ = operator.inv

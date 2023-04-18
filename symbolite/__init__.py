"""
    symbolite
    ~~~~~~~~~

    A minimal symbolic python package.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from .abstract import Function, Scalar, Symbol, Vector, scalar, vector

__all__ = ["Symbol", "Function", "Scalar", "scalar", "Vector", "vector"]

"""
    symbolite.core.base
    ~~~~~~~~~~~~~~~~~~~

    Symbolite base classes and functions.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import dataclasses
import typing as ty


class Unsupported(ValueError):
    """Label unsupported"""


class BaseCall:
    """Base class for calls"""

    name: str


class BaseFunction:
    """Base class for functions"""

    name: str
    namespace: str


class BaseSymbol:
    """Base class for symbols"""

    name: str
    namespace: str


@dataclasses.dataclass(frozen=True)
class Named:
    """A named primitive."""

    name: str
    namespace: ty.ClassVar = ""

    def __str__(self):
        if self.namespace:
            return self.namespace + "." + self.name

        return self.name

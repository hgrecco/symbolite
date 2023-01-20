"""
    symbolite.mappers
    ~~~~~~~~~~~~~~~~~

    Convenience mappers to manipulate expressions.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import collections
import typing
from typing import Union

from .operands import Function, Named

#####################
# Typing annotations
#####################

K = typing.TypeVar("K")
V = typing.TypeVar("V")


class InstanceGetItem(typing.Protocol[K, V]):
    def __getitem__(self, key: K) -> V:
        pass


class ClassGetItem(typing.Protocol[K, V]):
    def __class_getitem__(self, key: K) -> V:
        pass


GetItem = Union[InstanceGetItem, ClassGetItem]


class Unsupported(ValueError):
    """Label unsupported"""


class ToNameMapper:
    """Maps a Function, Value or Scalar to its name.

    A prefix can be added.
    """

    def __getitem__(self, item) -> str:
        if isinstance(item, Named):
            return str(item)
        raise KeyError(item)


##################
# Useful operands
##################


class MatchByName:
    def __init__(self, mapping: dict[str, typing.Any]):
        self.mapping = mapping

    def __getitem__(self, item) -> str:
        if isinstance(item, Named):
            return self.mapping[str(item)]
        raise KeyError(item)


class AsStr:
    def __class_getitem__(cls, item):
        if isinstance(item, Function):
            return item.format
        raise KeyError(item)


class IdentityMapper:
    def __class_getitem__(cls, item):
        return item


class CaptureCount:
    def __init__(self):
        self.content = collections.Counter()

    def __getitem__(self, item):
        self.content.update((item,))
        return item


default_to_name_mapper = ToNameMapper()

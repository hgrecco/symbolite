"""
symbolite.core.named
~~~~~~~~~~~~~~~~~~~~

Provides name, namespace for other objects.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Generator

from .operations import yield_named
from .util import repr_without_defaults


@dataclasses.dataclass(frozen=True, repr=False)
class Named:
    name: str | None = None
    namespace: str = ""

    def __str__(self) -> str:
        if self.name is None:
            return "<anonymous>"

        if self.namespace:
            return self.namespace + "." + self.name

        return self.name

    def __repr__(self) -> str:
        return repr_without_defaults(self)

    def format(self, *args: Any, **kwargs: Any) -> str: ...


def filter_namespace(
    namespace: str | None = "", include_anonymous: bool = False
) -> Callable[[Named], bool]:
    def _inner(s: Named) -> bool:
        if namespace is None:
            return True
        return s.namespace == namespace

    return _inner


def symbol_namespaces(self: Any) -> set[str]:
    """Return a set of symbol libraries"""
    return set(map(lambda s: s.namespace, yield_named(self, False)))


def symbol_names(self: Any, namespace: str | None = "") -> set[str]:
    """Return a set of symbol names (with full namespace indication).

    Parameters
    ----------
    namespace: str or None
        If None, all symbols will be returned independently of the namespace.
        If a string, will compare Symbol.namespace to that.
        Defaults to "" which is the namespace for user defined symbols.
    """
    ff = filter_namespace(namespace)
    return set(map(str, filter(ff, yield_named(self, False))))


@yield_named.register
def _(self: Named, include_anonymous: bool = False) -> Generator[Named, None, None]:
    if include_anonymous or self.name is not None:
        yield self

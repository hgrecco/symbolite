"""
    symbolite.core.util
    ~~~~~~~~~~~~~~~~~~~

    Symbolite core util functions.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from types import ModuleType
from typing import Any, Callable, Hashable, Iterator, Mapping, TypeVar

from ..abstract import scalar
from . import evaluate, inspect, substitute

TH = TypeVar("TH", bound=Hashable)


def solve_dependencies(dependencies: Mapping[TH, set[TH]]) -> Iterator[set[TH]]:
    """Solve a dependency graph.

    Parameters
    ----------
    dependencies :
        dependency dictionary. For each key, the value is an iterable indicating its
        dependencies.

    Yields
    ------
    set
        iterator of sets, each containing keys of independents tasks dependent only of
        the previous tasks in the list.

    Raises
    ------
    ValueError
        if a cyclic dependency is found.
    """
    while dependencies:
        # values not in keys (items without dep)
        t = {i for v in dependencies.values() for i in v} - dependencies.keys()
        # and keys without value (items without dep)
        t.update(k for k, v in dependencies.items() if not v)
        # can be done right away
        if not t:
            raise ValueError(
                "Cyclic dependencies exist among these items: {}".format(
                    ", ".join(repr(x) for x in dependencies.items())
                )
            )
        # and cleaned up
        dependencies = {k: v - t for k, v in dependencies.items() if v}
        yield t


def compute_dependencies(
    content: Mapping[TH, Any],
    is_dependency: Callable[[Any], bool],
):
    dependencies = {}
    for k, v in content.items():
        contents = inspect(v)
        if contents == {k: 1}:
            dependencies[k] = set()
        else:
            dependencies[k] = set(filter(is_dependency, contents.keys()))
    return dependencies


def substitute_content(
    content: Mapping[TH, Any],
    *,
    is_dependency: Callable[[Any], bool],
) -> dict[TH, scalar.NumberT]:
    dependencies = compute_dependencies(content, is_dependency)
    layers = solve_dependencies(dependencies)

    out: dict[TH, Any] = {}
    for layer in layers:
        for item in layer:
            out[item] = substitute(content[item], out)

    return out


def eval_content(
    content: Mapping[TH, Any],
    *,
    libsl: ModuleType,
    is_dependency: Callable[[Any], bool],
) -> dict[TH, scalar.NumberT]:
    """Evaluate a group of

    Parameters
    ----------
    content
        a mapping of assigments.
    libsl
        symbolite implementation module.
    is_root
        callable that takes a python object/value and returns True
        if it should be considered as having no dependencies.
    is_dependency
        callable that takes a python object/value and returns True
        if it should be considered as the dependency of another.
    """
    dependencies = compute_dependencies(content, is_dependency)
    layers = solve_dependencies(dependencies)

    out: dict[TH, scalar.NumberT] = {}
    for layer in layers:
        for item in layer:
            out[item] = evaluate(substitute(content[item], out), libsl)

    return out

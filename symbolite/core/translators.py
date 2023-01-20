"""
    symbolite.translators
    ~~~~~~~~~~~~~~~~~~~~~

    Translate symbolic expressions to values and strings.

    :copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import collections
import importlib
import inspect as pyinspect
import pkgutil
import types
import typing as ty

from .finder import find_symbolite_packages
from .mappers import (
    AsStr,
    CaptureCount,
    GetItem,
    IdentityMapper,
    MatchByName,
    Unsupported,
    default_to_name_mapper,
)
from .operands import Call, Named, OperandMixin


def _modname_to_lib(module_name: str):
    return f"lib{module_name.split('.')[-1]}"


def get_lib_implementation(
    libname: str, filter_by_name: str | tuple[str, ...] | None = None
) -> dict[str, types.ModuleType]:
    """Get all implementations for a given lib

    Parameters
    ----------
    libname
        library name
    filter_by_name
        implementations that must be imported.
    """
    this_impl = importlib.import_module(f"symbolite.{libname}")
    out = {}
    if isinstance(filter_by_name, str):
        filter_by_name = (filter_by_name,)

    for n in pkgutil.iter_modules(this_impl.__path__):
        if n.name == "abstract":
            continue
        if filter_by_name and n.name not in filter_by_name:
            continue
        out[n.name] = importlib.import_module(f"symbolite.{libname}.{n.name}")
    return out


def get_implementations(
    filter_by_name: str | tuple[str, ...] | None = None
) -> dict[str, dict[str, types.ModuleType]]:
    """Get all implementations for every lib.

    Parameters
    ----------
    filter_by_name
        implementations that must be imported.

    """

    out = {}
    for m in find_symbolite_packages():
        name = m.split(".")[-1]
        out[f"lib{name}"] = get_lib_implementation(name, filter_by_name)

    return out


DEFAULT_IMPLS = {k: v["default"] for k, v in get_implementations("default").items()}


def find_libs_in_stack(expr: OperandMixin = None) -> dict[str, types.ModuleType]:
    """Find libraries in stack.

    Parameters
    ----------
    expr
        If None, an implementation for every abstract library
        will be look for.
        If an expression, it will be first inspected to find
        which libraries it is using and only those will be look for.

    """
    if expr is None:
        missing_libs = set(DEFAULT_IMPLS.keys())
    else:
        missing_libs = expr.symbol_namespaces() - set("")
    out = {}
    frame = pyinspect.currentframe().f_back
    while frame and missing_libs:
        for key in set(
            missing_libs
        ):  # we create a copy to be able to modify the original.
            if key in frame.f_locals:
                out[key] = frame.f_locals[key]
                missing_libs.remove(key)
        frame = frame.f_back

    return out


_default_str_mapper = collections.ChainMap(AsStr, default_to_name_mapper)


def map_expression(expr: OperandMixin, mapper: GetItem[Named, ty.Any]):
    """Map each a symbol recursively.

    Parameters
    ----------
    expr
        symbolic expression.
    mapper
        mapping from symbols to other objects, using getitem.
    """

    if isinstance(expr, Call):
        args = tuple(map_expression(arg, mapper) for arg in expr.args)
        kwargs = {k: map_expression(arg, mapper) for k, arg in expr.kwargs_items}

        f = mapper[expr.func]

        if f is Unsupported:
            raise Unsupported(f"{expr.func} is not supported by this mapper")

        return f(*args, **kwargs)

    if isinstance(expr, Named):
        return mapper[expr]

    return expr


def map_expression_by_attr(expr: OperandMixin, **libs: types.ModuleType):
    """Map each a symbol recursively.

    Parameters
    ----------
    expr
        symbolic expression.
    libs
        mapping from symbols to other objects, using getattr.
    """

    if isinstance(expr, Call):
        # first get dispatch by symbolite name
        args = tuple(map_expression_by_attr(arg, **libs) for arg in expr.args)
        kwargs = {
            k: map_expression_by_attr(arg, **libs) for k, arg in expr.kwargs_items
        }

        # then get actual implementation
        f = getattr(libs[expr.func.namespace], expr.func.name)

        if f is Unsupported:
            raise Unsupported(f"{expr.func} is not supported by this implementation")

        return f(*args, **kwargs)

    if isinstance(expr, Named):

        if expr.namespace != "":  # not user defined symbol
            return getattr(libs[expr.namespace], expr.name)

        cls = getattr(
            libs[_modname_to_lib(expr.__class__.__module__)], expr.__class__.__name__
        )
        if cls is Unsupported:
            raise Unsupported(
                f"{expr.__class__.__name__} is not supported by this implementation"
            )

        return cls(expr.name)

    return expr


def inspect(expr: OperandMixin):
    """Inspect an expression and return what is there.
    and within each key there is a dictionary relating the
    given object with the number of times it appears.

    Parameters
    ----------
    expr
        symbolic expression.
    """

    c = CaptureCount()
    map_expression(expr, c)
    return c.content


def replace(expr: OperandMixin, *mapers):
    """Replace symbols, functions, values, etc by others.

    If multiple mappers are provided,
        they will be used in order (using a ChainMap)

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    expr
        symbolic expression.
    *mappers
        dictionaries mapping source to destination objects.
    """

    return map_expression(expr, collections.ChainMap(*mapers, IdentityMapper))


def replace_by_name(expr: OperandMixin, **symbols):
    """Replace Symbols by values or objects, matching by name.

    If a given object is not found in the mappers,
        the same object will be returned.

    Parameters
    ----------
    expr
        symbolic expression.
    **symbols
        keyword arguments connecting names to values.
    """

    mapper = MatchByName(symbols)
    return map_expression(expr, collections.ChainMap(mapper, IdentityMapper))


def evaluate(
    expr: OperandMixin,
    **libs: types.ModuleType,
):
    """Evaluate expression.

    Parameters
    ----------
    expr
        symbolic expression.
    libs
        implementation module
    """

    libs = {**DEFAULT_IMPLS, **libs}

    return map_expression_by_attr(expr, **libs)


def as_string(
    expr: OperandMixin,
    mapper: GetItem[OperandMixin, str] = _default_str_mapper,
):
    """Evaluate a call or symbol into a value.

    Parameters
    ----------
    expr
        symbolic expression.
    mapper
        maps user defined symbol. to values.
        defaults to using the same name.
    """
    return map_expression(expr, mapper)


def as_function(
    expr: OperandMixin,
    function_name: str,
    params: tuple[str, ...],
    **libs: types.ModuleType,
):
    """Converts the expression to a callable function.

    Parameters
    ----------
    expr
        symbolic expression.
    function_name
        name of the function to be used.
    params
        names of the parameters.s
    libs:
        implementation module
    """

    function_def = (
        f"""def {function_name}({", ".join(params)}): return {as_string(expr)}"""
    )

    libs = {**DEFAULT_IMPLS, **libs}

    lm = {}
    exec(function_def, libs, lm)
    return lm[function_name]

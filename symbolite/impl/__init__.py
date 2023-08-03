import importlib
import inspect
import types
from pathlib import Path


def find_module_in_stack(name: str = "libsl") -> types.ModuleType | None:
    """Find libraries in stack.

    Parameters
    ----------
    expr
        If None, an implementation for every abstract library
        will be look for.
        If an expression, it will be first inspected to find
        which libraries it is using and only those will be look for.

    """
    frame = inspect.currentframe()
    while frame:
        if name in frame.f_locals:
            mod = frame.f_locals[name]
            if mod is not None:
                return mod
        frame = frame.f_back

    return None


def get_all_implementations() -> dict[str, types.ModuleType]:
    out = {}

    path = Path(__file__)
    for p in path.parent.iterdir():
        name = p.stem
        if name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f".{name}", package=__package__)
        except ImportError:
            pass
        else:
            out[name] = module

    return out

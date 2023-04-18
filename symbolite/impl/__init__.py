import inspect
import types


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
    from . import libstd

    out["libstd"] = libstd

    try:
        from . import libnumpy

        out["libnumpy"] = libnumpy
    except ImportError:
        pass

    try:
        from . import libstd

        out["libstd"] = libstd
    except ImportError:
        pass

    return out

import types

import pytest

from symbolite import UserFunction, evaluate
from symbolite.impl import get_all_implementations

all_impl = get_all_implementations()


def f_1_1(x):
    return 2 * x


def f_2_1(x, y):
    return x + y


def f_1_2(x):
    return 1 * x, 2 * x


@pytest.mark.parametrize(
    "func,args,result",
    [
        (f_1_1, (1,), 2),
        (f_2_1, (1, 2), 3),
        (f_1_2, (1,), (1, 2)),
    ],
)
@pytest.mark.parametrize("libsl", all_impl.values(), ids=all_impl.keys())
def test_user_functions(func, args, result, libsl: types.ModuleType):
    uf = UserFunction.from_function(func)

    assert evaluate(uf(*args), libsl) == result

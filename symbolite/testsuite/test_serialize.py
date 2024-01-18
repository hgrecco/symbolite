import pickle

from pytest import mark

from .. import Scalar, Symbol, Vector


@mark.parametrize("cls", [Scalar, Symbol, Vector])
def test_pickle(cls):
    obj = cls()
    dump = pickle.dumps(obj)
    load = pickle.loads(dump)
    assert load == obj

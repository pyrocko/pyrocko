
from pyrocko import guts


def load(path, want=None):
    return [
        obj for obj in guts.load_all(filename=path)
        if want is None or isinstance(obj, want)]


__all__ = [
    'load',
]

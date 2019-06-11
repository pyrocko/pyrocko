
from pyrocko import gf
from .error import CannotCreatePath


def remake_dir(dpath, force):
    try:
        return gf.store.remake_dir(dpath, force)

    except gf.CannotCreate as e:
        raise CannotCreatePath(str(e))

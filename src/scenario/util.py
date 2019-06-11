
from pyrocko import gf
from .error import CannotCreate


def remake_dir(dpath, force):
    try:
        return gf.store.remake_dir(dpath, force)

    except gf.CannotCreate as e:
        raise CannotCreate(str(e))

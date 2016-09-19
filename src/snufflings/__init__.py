from pyrocko.snufflings import minmax, rms, stalta, geofon, ampspec, \
    catalogs, download
modules = [minmax, rms, download, stalta, geofon, ampspec, catalogs]


def __snufflings__():
    snufflings = []
    for mod in modules:
        snufflings.extend(mod.__snufflings__())

    for snuffling in snufflings:
        snuffling.setup()
        snuffling.set_name(snuffling.get_name() + ' (builtin)')

    return snufflings

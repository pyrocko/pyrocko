
import os
from os.path import expanduser, expandvars
from copy import deepcopy

from pyrocko import util
from pyrocko.guts import Object, Float, String, load, dump, List, Dict

guts_prefix = 'pf'

pyrocko_dir_tmpl = os.environ.get(
    'PYROCKO_DIR',
    os.path.join('~', '.pyrocko'))

conf_path_tmpl = os.path.join(pyrocko_dir_tmpl, 'config.pf')
default_phase_key_mapping = {
    'F1': 'P', 'F2': 'S', 'F3': 'R', 'F4': 'Q', 'F5': '?'}


class BadConfig(Exception):
    pass


class PathWithPlaceholders(String):
    '''Path, possibly containing placeholders.'''
    pass


class PyrockoConfig(Object):
    cache_dir = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'cache'))
    earthradius = Float.T(default=6371.*1000.)
    gf_store_dirs = List.T(PathWithPlaceholders.T())
    gf_store_superdirs = List.T(PathWithPlaceholders.T())
    phase_key_mapping = Dict.T(
        String.T(), String.T(), default=default_phase_key_mapping)
    visible_length_default = String.T(default='Short')
    topo_dir = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'topo'))
    geonames_dir = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'geonames'))
    leapseconds_path = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'leap-seconds.list'))
    leapseconds_url = String.T(
        default='http://www.ietf.org/timezones/data/leap-seconds.list')

    @property
    def visible_length_options(self):
        return {'Short': 6000, 'Medium': 20000, 'Long': 60000}

    def get_phase_name(self, key):
        return self.phase_key_mapping.get('F%s' % key, 'Undefined')


def expand(x):
    x = expanduser(expandvars(x))
    return x


def rec_expand(x):
    for prop, val in x.T.ipropvals(x):
        if prop.multivalued:
            for i, ele in enumerate(val):
                if isinstance(prop.content_t, PathWithPlaceholders.T):
                    newele = expand(ele)
                    if newele != ele:
                        val[i] = newele

                elif isinstance(ele, Object):
                    rec_expand(ele)
        else:
            if isinstance(prop, PathWithPlaceholders.T):
                newval = expand(val)
                if newval != val:
                    setattr(x, prop.name, newval)

            elif isinstance(val, Object):
                rec_expand(val)


def processed(config):
    config = deepcopy(config)
    rec_expand(config)
    return config


def mtime(p):
    return os.stat(p).st_mtime

g_conf_mtime = None
g_conf = None


def raw_config():
    global g_conf
    global g_conf_mtime

    conf_path = expand(conf_path_tmpl)

    if not os.path.exists(conf_path):
        g_conf = PyrockoConfig()
        write_config(g_conf)

    conf_mtime_now = mtime(conf_path)
    if conf_mtime_now != g_conf_mtime:
        g_conf = load(filename=conf_path)
        if not isinstance(g_conf, PyrockoConfig):
            raise BadConfig('config file does not contain a '
                            'valid "pf.PyrockoConfig" section.')

        g_conf_mtime = conf_mtime_now

    return g_conf


def config():
    return processed(raw_config())


def write_config(conf):
    conf_path = expand(conf_path_tmpl)
    util.ensuredirs(conf_path)
    dump(conf, filename=conf_path)

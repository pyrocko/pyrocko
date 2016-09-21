
import os
from os.path import expanduser, expandvars
from copy import deepcopy

from pyrocko import util
from pyrocko.guts import Object, Float, String, load, dump, List, Dict, TBase

guts_prefix = 'pf'

pyrocko_dir_tmpl = os.environ.get(
    'PYROCKO_DIR',
    os.path.join('~', '.pyrocko'))

conf_path_tmpl = os.path.join(pyrocko_dir_tmpl, 'config{module}.pf')

default_phase_key_mapping = {
    'F1': 'P', 'F2': 'S', 'F3': 'R', 'F4': 'Q', 'F5': '?'}


class BadConfig(Exception):
    pass


class PathWithPlaceholders(String):
    '''Path, possibly containing placeholders.'''
    pass


class VisibleLengthSetting(Object):
    class __T(TBase):
        def regularize_extra(self, val):
            if isinstance(val, list):
                return self.cls(key=val[0], value=val[1])

            return val

        def to_save(self, val):
            return (val.key, val.value)

        def to_save_xml(self, val):
            raise NotImplementedError()

    key = String.T()
    value = Float.T()


class SnufflerConfig(Object):
    visible_length_setting = List.T(
        VisibleLengthSetting.T(),
        default=[VisibleLengthSetting(key='Short', value=6000.),
                 VisibleLengthSetting(key='Medium', value=20000.),
                 VisibleLengthSetting(key='Long', value=60000.)])
    phase_key_mapping = Dict.T(
        String.T(), String.T(), default=default_phase_key_mapping)

    def get_phase_name(self, key):
        return self.phase_key_mapping.get('F%s' % key, 'Undefined')


class PyrockoConfig(Object):
    cache_dir = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'cache'))
    earthradius = Float.T(default=6371.*1000.)
    gf_store_dirs = List.T(PathWithPlaceholders.T())
    gf_store_superdirs = List.T(PathWithPlaceholders.T())
    topo_dir = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'topo'))
    geonames_dir = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'geonames'))
    leapseconds_path = PathWithPlaceholders.T(
        default=os.path.join(pyrocko_dir_tmpl, 'leap-seconds.list'))
    leapseconds_url = String.T(
        default='http://www.ietf.org/timezones/data/leap-seconds.list')


def fill_template(tmpl, config_type):
    tmpl = tmpl .format(
        module=('.' + config_type) if config_type != 'pyrocko' else '')
    return tmpl


def expand(x):
    x = expanduser(expandvars(x))
    return x


def rec_expand(x):
    for prop, val in x.T.ipropvals(x):
        if prop.multivalued:
            if val is not None:
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

configs = {'pyrocko': PyrockoConfig,
           'snuffler': SnufflerConfig}


def raw_config(config_type):
    global g_conf
    global g_conf_mtime

    conf_path = expand(conf_path_tmpl)
    conf_path = fill_template(conf_path, config_type)

    if not os.path.exists(conf_path):
        g_conf = configs[config_type]()
        write_config(g_conf, config_type)

    conf_mtime_now = mtime(conf_path)
    if conf_mtime_now != g_conf_mtime:
        g_conf = load(filename=conf_path)
        if not isinstance(g_conf, configs[config_type]):
            raise BadConfig('config file does not contain a '
                            'valid {config_cls} section.'
                            .format(configs[config_type].__class__.__name__))

        g_conf_mtime = conf_mtime_now

    return g_conf


def config(config_type='pyrocko'):
    return processed(raw_config(config_type))


def write_config(conf, config_type):
    conf_path = expand(conf_path_tmpl)
    conf_path = fill_template(conf_path, config_type)
    util.ensuredirs(conf_path)
    dump(conf, filename=conf_path)

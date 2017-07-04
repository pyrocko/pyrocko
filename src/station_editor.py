from collections import OrderedDict
import PyQt4.QtCore as qc
import PyQt4.QtGui as qg
import logging
import time

logger = logging.getLogger('pyrocko.station_editor')


def qsi(key='', str_format='{}'):
    return qg.QStandardItem(str_format.format(key))


class StationModel(qg.QStandardItemModel):

    ''' Station data container'''
    def __init__(self, *args, **kwargs):
        qg.QStandardItemModel.__init__(self, *args, **kwargs)
        self.setHorizontalHeaderLabels(
            ['NSL', 'Lat', 'Lon', 'Depth', 'Name',
            'Channel', 'Sampling Rate', 'Azimuth', 'Dip', 'Gain'])

        self.rows = []
        self.parents = {}
        self.nsl_parents = {}
        self.channel_nodes = {}

    def get_parent_node(self, key):
        ''' Create a new parent node if required and add it to model or return
        the existing node.'''
        node = self.parents.get(key, False)
        if not node:
            node = qg.QStandardItem(key)
            self.parents[key] = node
            self.appendRow(node)

        return node

    def add_stations(self, stations):
        ''' Add stations shown in view.

        :param stations: list of py:class:`pyrocko.model.Station`'''

        for nsl, station in stations.items():
            channels = station.channels
            network_parent = self.get_parent_node(nsl[0])

            nsl_item = '.'.join(nsl)

            row = [nsl_item, station.lat, station.lon,
                station.depth-station.elevation, station.name]
            if row not in self.rows:
                self.rows.append(row)
                row = map(qsi, row)
                network_parent.appendRow(row)

            self.nsl_parents[nsl] = row

            for c in channels:
                self.set_channel_data(
                    (nsl + (c.name, )),
                    azimuth=c.azimuth,
                    dip=c.dip,
                    gain=c.gain)

    def set_channel_data(self, nslc, **kwargs):
        ''' Set data shown for channel of *nslc*.

        :param kwargs: dict with attributes of
            :py:class:`pyrocko.model.Channel`
        '''
        parent = self.nsl_parents.get(nslc[:3], False)
        if not parent:
            # Na station for given channel
            return

        channel_node = self.channel_nodes.get(nslc, False)
        if not channel_node:
            kwargs['channel'] = nslc[-1]
            channel_node = ChannelData(**kwargs)
            self.channel_nodes[nslc] = channel_node
            parent[0].appendRow(channel_node.make_node())
        else:
            channel_node.update(**kwargs)


class ChannelData():
    def __init__(self, *args, **kwargs):
        self.data = OrderedDict(
            [('_1', ''), ('_2', ''), ('_3', ''), ('_4', ''), ('_5', ''),
            ('channel', ''), ('sampling_rate', ''), ( 'azimuth', ''),
             ('dip', ''), ('gain', '')])
        self.data.update(kwargs)

        for k, v in self.data.items():
            self.data[k] = qsi(v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].setData(v, 0)

    def make_node(self):
        return self.data.values()


class StationTreeView(qg.QTreeView):
    def __init__(self):
        qg.QTreeView.__init__(self)

        self.setSortingEnabled(True)
        self.setSelectionBehavior(qg.QAbstractItemView.SelectRows)
        self.setUniformRowHeights(True)
        self.setFirstColumnSpanned(0, self.rootIndex(), True)
        self.setSizePolicy(qg.QSizePolicy.Minimum, qg.QSizePolicy.Minimum)


class StationEditor(qg.QFrame):

    def __init__(self, *args, **kwargs):
        qg.QFrame.__init__(self, *args, **kwargs)
        self.view = StationTreeView()
        layout = qg.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.view)

        self.station_model = StationModel()
        self.view.setModel(self.station_model)

    def on_pile_changed(self):
        d = self.viewer.pile.nslc_id_deltats
        for (nslc, deltat), count in d.items():
            self.station_model.set_channel_data(nslc, sampling_rate=1./deltat)

    def on_stations_added(self):
        self.station_model.add_stations(self.viewer.stations)
        self.view.expandToDepth(0)

    def set_viewer(self, viewer):
        '''Set a pile_viewer and connect to signals.'''
        self.viewer = viewer
        self.connect(viewer, qc.SIGNAL('stations_added()'),
                     self.on_stations_added)
        self.connect(viewer, qc.SIGNAL('pile_has_changed_signal()'),
                     self.on_pile_changed)

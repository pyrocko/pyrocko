import numpy as num
from collections import defaultdict
from PyQt4 import QtCore as qc
from PyQt4 import QtGui as qg

from pyrocko.gui_util import EventMarker, PhaseMarker, make_QPolygonF
from pyrocko.beachball import mt2beachball, BeachballError
from pyrocko import orthodrome, moment_tensor
from pyrocko.plot import tango_colors
import logging

logger = logging.getLogger('pyrocko.station_editor')

qvar = qc.QVariant


class TreeItem(object):
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.childItems = []

    def appendChild(self, item):
        self.childItems.append(item)

    def child(self, row):
        return self.childItems[row]

    def childCount(self):
        return len(self.childItems)

    def columnCount(self):
        return len(self.itemData)

    def rowCount(self):
        return 1

    def data(self, column):
        try:
            return self.itemData[column]
        except IndexError as e:
            return None

    def parent(self):
        return self.parentItem

    def row(self):
        if self.parentItem:
            return self.parentItem.childItems.index(self)

        return 0

    def setData(self, data):
        self.itemData = data


class TreeModel(qc.QAbstractItemModel):
    def __init__(self, data=None, parent=None):
        super(TreeModel, self).__init__(parent)
        self.parents = []
        self.rootItem = TreeItem(['NSL', 'Lat', 'Lon', 'Name', 'deltat', 'Azi', 'Dip', 'Gain'])
        if data:
            self.setupModelData(data, parent=self.rootItem)

    def setData(self, index, value, role):
       if index.isValid() and role == qc.Qt.EditRole:

           prev_value = self.getValue(index)

           item = index.internalPointer()

           item.setData(unicode(value.toString()))

           return True
       else:
           return False

    def removeRows(self, position=0, count=1,  parent=qc.QModelIndex()):

       node = self.nodeFromIndex(parent)
       self.beginRemoveRows(parent, position, position + count - 1)
       node.childItems.pop(position)
       self.endRemoveRows()

    def nodeFromIndex(self, index):
       if index.isValid():
           return index.internalPointer()
       else:
           return self.rootItem

    def getValue(self, index):
       item = index.internalPointer()
       return item.data(index.column())

    def data(self, index, role):
       if not index.isValid():
           return None
       if role != qc.Qt.DisplayRole:
           return None

       item = index.internalPointer()
       return qc.QVariant(item.data(index.column()))

    def flags(self, index):
       if not index.isValid():
           return qc.Qt.NoItemFlags

       return qc.Qt.ItemIsEnabled | qc.Qt.ItemIsSelectable | qc.Qt.ItemIsEditable

    def headerData(self, section, orientation, role):
        if orientation == qc.Qt.Horizontal and role == qc.Qt.DisplayRole:
            return qc.QVariant(self.rootItem.data(section))

        return None

    def index(self, row, column, parent):

        if row < 0 or column < 0 or row >= self.rowCount(parent) or column >= self.columnCount(parent):
            return qc.QModelIndex()

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return qc.QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return qc.QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self.rootItem:
            return qc.QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def columnCount(self, parent):
       if parent.isValid():
           return parent.internalPointer().columnCount()
       else:
           return self.rootItem.columnCount()

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def setupModelData(self, stations, deltats=None, parent=None):
        if parent:
            self.parents.append(parent)

        deltats = deltats or {}

        for sk, station in stations.items():
            channels = station.get_channels()
            nsl = station.nsl()
            nsl_str = '.'.join(station.nsl())
            row_data = [
                qvar(nsl_str), qvar(station.lat), qvar(station.lon),
                qvar(''), qvar(''), qvar(''), qvar(''), qvar('')]
            columnData = station.nsl()
            item = TreeItem(row_data, self.parents[-1])
            self.parents[-1].appendChild(item)

            self.parents.append(self.parents[-1].child(
                self.parents[-1].childCount() - 1))

            channel_items = []
            if channels:
                for channel in channels:
                    dt = ''
                    has_dts = deltats.get(nsl, None)
                    if has_dts:
                        dt = has_dts.get(channel.name, None)

                    d = [ qvar(nsl_str), qvar(''), qvar(''), qvar(dt),
                         qvar(channel.name), qvar(channel.azimuth),
                         qvar(channel.dip), qvar(channel.gain), ]
                    channel_items.append(TreeItem(d, self.parents[-1]))

            else:
                print 'IMPLEMENT'

            for channel_item in channel_items:
                self.parents[-1].appendChild(channel_item)

            if len(self.parents) > 0:
                self.parents.pop()


class StationEditor(qg.QFrame):

    def __init__(self, *args, **kwargs):
        qg.QFrame.__init__(self, *args, **kwargs)
        layout = qg.QVBoxLayout()
        self.station_tree_view = qg.QTreeView(self)
        self.station_tree_view.setSortingEnabled(True)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.station_tree_view)
        self.station_tree_view.setSizePolicy(qg.QSizePolicy.Minimum,
                           qg.QSizePolicy.Minimum)

    def update_contents(self):
        stations = self.pile_viewer.stations
        deltats = defaultdict(dict)
        for tr in self.pile_viewer.pile.iter_traces(load_data=False):
            # can change!
            nslc = tr.nslc_id
            deltats[nslc[:3]][nslc[-1]] = tr.deltat

        station_model = TreeModel(stations, self)
        station_model.setupModelData(stations, deltats, self.root_item)
        self.station_tree_view.setModel(station_model)

    def set_viewer(self, viewer):
        '''Set a pile_viewer and connect to signals.'''
        self.pile_viewer = viewer
        self.connect(viewer, qc.SIGNAL('stationsAdded()'),
                     self.update_contents)
        self.connect(viewer, qc.SIGNAL('pile_has_changed_signal()'),
                     self.update_contents)
        self.root_item = TreeItem([])

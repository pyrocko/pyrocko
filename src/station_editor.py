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
        self.show_details = lambda x: True

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

    def mouseDoubleClickEvent(self, event):
        self.show_details(self)


class TreeModel(qc.QAbstractItemModel):
    def __init__(self, parent=None, root_item=None):
        super(TreeModel, self).__init__(parent)
        self.items = {}
        #self.parents = []
        self.parents = {}
        if root_item is not None:
            self.parents = {'root': root_item}
            #self.parents = [root_item]
        #self.rootItem = TreeItem(['NSL', 'Lat', 'Lon', 'Name', 'deltat', 'Azi', 'Dip', 'Gain'])
        #if data:
        #    self.setupModelData(data, parent=self.rootItem)

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

    def update_deltats(self, deltats):
        for nslc_dt, n in deltats.items():
            nslc, dt = nslc_dt
            nsl = nslc[:3]
            parental_key = nslc[:3]
            #parent = self.parents.get(parental_key, self.parents['root'])
            parent = self.parents.get(parental_key, None)
            if parent is None:
                print nsl
                ''' NEED to add dummy station here'''
                continue

            has_dts = deltats.get(nsl, None)
            if has_dts:
                dt = has_dts.get(channel.name, None)

            #d = [qvar('.'.join(nsl)), qvar(''), qvar(''), qvar(dt),
            #     qvar(channel.name), qvar(channel.azimuth),
            #     qvar(channel.dip), qvar(channel.gain), ]
            d = [qvar('.'.join(nsl)), qvar(''), qvar(''), qvar(dt),
                 qvar(''), qvar('keeeeep'),
                 qvar(''), qvar(''), ]
            #channel_items.append(TreeItem(d, self.parents[-1]))
            #channel_items.append(TreeItem(d, parent))

            #self.parents[-1].appendChild(channel_item)
            ##if not parent_isroot:
            parent.appendChild(d)
            self.parents[parental_key] = parent.child(
                #parent.childCount() - 1))
                parent.childCount() - 1)


    def update_stations(self, stations=None):
        #if parent:
        #    self.parents.append(parent)
        print stations
        for (n, s), station in stations.items():
            nsl_str = '.'.join(station.nsl())
            parental_key = nsl_str
            #parent = self.parents.get(parental_key, self.parents['root'])
            parent = self.parents.get(parental_key, None)
            if parent is None:
                parent = self.parents['root']
                parent_isroot = True
            else:
                parent_isroot = False
            channel_items = []
            if station is not None:
                nsl = station.nsl()
                lat = station.lat
                lon = station.lon

                channels = station.get_channels()
                if channels:
                    for channel in channels:
                        dt = ''
                        d = [qvar(nsl_str), qvar(''), qvar(''), qvar(dt),
                             qvar(channel.name), qvar(channel.azimuth),
                             qvar(channel.dip), qvar(channel.gain), ]
                        #channel_items.append(TreeItem(d, self.parents[-1]))
                        channel_items.append(TreeItem(d, parent))

                else:
                    print 'IMPLEMENT'

                for channel_item in channel_items:
                    #self.parents[-1].appendChild(channel_item)
                    ##if not parent_isroot:
                    parent.appendChild(channel_item)

            else:
                lat = ''
                lon = ''

            row_data = [
                qvar(nsl_str), qvar(lat), qvar(lon),
                qvar(''), qvar(''), qvar(''), qvar(''), qvar('')]
            print row_data
            item = TreeItem(row_data, parent)
            if not parent_isroot:
                parent.appendChild(item)

            #self.parents.append(self.parents[-1].child(
            self.parents[parental_key] = parent.child(
                #parent.childCount() - 1))
                parent.childCount() - 1)

            if len(self.parents) > 0:
                self.parents.pop(parental_key)


class StationEditor(qg.QFrame):

    def __init__(self, *args, **kwargs):
        qg.QFrame.__init__(self, *args, **kwargs)
        self.station_tree_view = qg.QTreeView(self)
        self.station_tree_view.setSortingEnabled(True)

        layout = qg.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.station_tree_view)
        self.root_item = TreeItem(['NSL', 'Lat', 'Lon', 'Name', 'deltat', 'Azi', 'Dip', 'Gain'])

        self.station_tree_view.setSizePolicy(qg.QSizePolicy.Minimum,
                           qg.QSizePolicy.Minimum)
        self.station_model = TreeModel(self, root_item=self.root_item)
        #self.station_model.rootItem = self.root_item
        #self.station_model.setupModelData([], )

    #def update_contents(self):
    #    stations = self.pile_viewer.stations
    #    print stations
    #    print 'deltats', deltats
    #    self.station_tree_view.setModel(self.station_model)

    def on_pile_changed(self):
        self.station_model.update_deltats(
            deltats=self.pile_viewer.pile.nslc_id_deltats)

    def on_stations_added(self):
        self.station_model.update_stations(
            stations=self.pile_viewer.stations)

    def set_viewer(self, viewer):
        '''Set a pile_viewer and connect to signals.'''
        self.pile_viewer = viewer
        self.connect(viewer, qc.SIGNAL('stationsAdded()'),
                     self.on_stations_added)
        self.connect(viewer, qc.SIGNAL('pile_has_changed_signal()'),
                     self.on_pile_changed)

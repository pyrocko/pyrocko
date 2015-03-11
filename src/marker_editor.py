
from PyQt4.QtCore import *  # noqa
from PyQt4.QtGui import *  # noqa

from pyrocko.gui_util import EventMarker, PhaseMarker
from pyrocko import util


class MarkerTable(QTableView):
    def __init__(self, *args, **kwargs):
        QTableView.__init__(self, *args, **kwargs)

        self.setSelectionBehavior(QAbstractItemView.SelectRows)


class MarkerModel(QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        QAbstractTableModel.__init__(self, *args, **kwargs)
        self.viewer = None

    def set_viewer(self, viewer):
        self.viewer = viewer
        self.connect(self.viewer, SIGNAL('markers_changed(int,int)'), self.markers_changed)

    def rowCount(self, parent):
        if not self.viewer:
            return 0
        return len(self.viewer.markers)

    def columnCount(self, parent):
        return 2

    def markers_changed(self, istart, istop):
        self.beginInsertRows(QModelIndex(), istart, istop-1)
        self.endInsertRows()

    def data(self, index, role):
        if not self.viewer:
            return QVariant()

        if role == Qt.DisplayRole:
            imarker = index.row()
            marker = self.viewer.markers[imarker]
            if index.column() == 0:
                if isinstance(marker, EventMarker):
                    s = 'E'
                elif isinstance(marker, PhaseMarker):
                    s = 'P'
                else:
                    s = ''

            if index.column() == 1:
                s = util.time_to_str(marker.tmin)

            return QVariant(QString(s))

        return QVariant()


class MarkerEditor(QFrame):
    def __init__(self, *args, **kwargs):
        QFrame.__init__(self, *args, **kwargs)

        layout = QGridLayout()
        self.setLayout(layout)
        self.marker_table = MarkerTable()
        self.marker_model = MarkerModel()
        self.marker_table.setModel(self.marker_model)
        layout.addWidget(self.marker_table, 0, 0)

    def set_viewer(self, viewer):
        self.marker_model.set_viewer(viewer)

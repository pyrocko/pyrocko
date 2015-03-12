
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
        self.headerdata = ['Type', 'Time', 'Mag']

    def set_viewer(self, viewer):
        self.viewer = viewer
        self.connect(self.viewer, SIGNAL('markers_changed(int,int)'), self.markers_changed)

    def rowCount(self, parent):
        if not self.viewer:
            return 0
        return len(self.viewer.markers)

    def columnCount(self, parent):
        return 3

    def markers_changed(self, istart, istop):
        self.beginInsertRows(QModelIndex(), istart, istop-1)
        self.endInsertRows()

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headerdata[col])
        else:
            return QVariant()

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


            if index.column() == 2:
                if isinstance(marker, EventMarker):
                    s = str(marker.get_event().magnitude)
                else:
                    s = ''

            return QVariant(QString(s))

        return QVariant()

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            imarker = index.row()
            marker = self.viewer.markers[imarker]
            if index.column() == 3:
                marker.get_event().magnitude = value

    def flags(self, index):
        if index.column() == 2:
            return Qt.ItemFlags(35)
        return Qt.ItemFlags(33)



class MarkerEditor(QFrame):
    def __init__(self, *args, **kwargs):
        QFrame.__init__(self, *args, **kwargs)

        layout = QGridLayout()
        self.setLayout(layout)
        self.marker_table = MarkerTable()

        self.marker_model = MarkerModel()
        self.marker_table.setModel(self.marker_model)
        self.selection_model = QItemSelectionModel(self.marker_model)
        self.marker_table.setSelectionModel(self.selection_model)
        self.connect(
            self.selection_model,
            SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
            self.set_selected_markers)

        layout.addWidget(self.marker_table, 0, 0)

        self.viewer = None

    def set_viewer(self, viewer):
        self.marker_model.set_viewer(viewer)
        self.viewer = viewer
        self.connect(self.viewer, SIGNAL('changed_marker_selection'), self.update_selection_model)

    def set_selected_markers(self, selected, deselected):
        ''' set markers selected in viewer at selection in table.'''
        selected_markers = [self.viewer.markers[i.row()] for i in self.selection_model.selectedRows()]
        self.viewer.deselect_all()
        self.viewer.set_selected_markers(selected_markers)

    def update_selection_model(self, indices):
        ''' :param indices: list of indices of selected markers.'''
        self.selection_model.clearSelection()
        for i in indices:
            left = self.marker_model.index(i, 0)
            right = self.marker_model.index(i, 2)
            row_selection = QItemSelection(left, right)
            row_selection.select(left, right)
            self.selection_model.select(row_selection, QItemSelectionModel.SelectionFlags(2))

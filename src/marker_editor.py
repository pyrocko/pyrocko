import operator
from PyQt4.QtCore import *  # noqa
from PyQt4.QtGui import *  # noqa

from pyrocko.gui_util import EventMarker, PhaseMarker
from pyrocko import util

_header_data = ['Type', 'Time', 'Magnitude']
_column_mapping = dict(zip(_header_data, range(len(_header_data))))
_attr_mapping = dict(zip([0,1,2], ['__class__', 'tmin', 'magnitude']))

class MarkerItemDelegate(QStyledItemDelegate):
    def __init__(self, *args, **kwargs):
        QStyledItemDelegate.__init__(self, *args, **kwargs)

    def initStyleOption(self, option, index):
        if not index.isValid():
            return

        QStyledItemDelegate.initStyleOption(self, option, index)
        if index.row()%2==0:
            option.backgroundBrush = QBrush(QColor(50,10,10,30))

class MarkerTable(QTableView):
    def __init__(self, *args, **kwargs):
        QTableView.__init__(self, *args, **kwargs)

        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSortingEnabled(True)

        self.setShowGrid(False)
        self.verticalHeader().hide()
        self.viewer = None

    def keyPressEvent(self, key_event):
        keytext = str(key_event.text())
        self.viewer.keyPressEvent(key_event)

    def resizeEvent(self, event):
        width = event.size().width()
        self.setColumnWidth(_column_mapping['Type'], width*0.1)
        self.setColumnWidth(_column_mapping['Time'], width*0.7)
        self.setColumnWidth(_column_mapping['Magnitude'], width*0.2)

    def set_viewer(self, viewer):
        self.viewer = viewer


class MarkerModel(QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        QAbstractTableModel.__init__(self, *args, **kwargs)
        self.viewer = None
        self.headerdata = _header_data

    def set_viewer(self, viewer):
        self.viewer = viewer
        self.connect(self.viewer, SIGNAL('markers_changed(int,int)'), self.markers_changed)

    def rowCount(self, parent):
        if not self.viewer:
            return 0
        return len(self.viewer.markers)

    def columnCount(self, parent):
        return len(_column_mapping)

    def markers_changed(self, istart, istop):
        self.beginInsertRows(QModelIndex(), istart, istop-1)
        self.endInsertRows()

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return QVariant(self.headerdata[col])
            elif role == Qt.SizeHintRole:
                return QSize(10,20)
        else:
            return QVariant()

    def sort(self, col, order):
        self.emit(SIGNAL("layoutAboutToBeChanged()"))
        if _attr_mapping[col]=='magnitude':
            markers = self.viewer.markers
            event_markers = []
            other = []
            for im, m in enumerate(markers):
                if isinstance(m, EventMarker):
                    event_markers.append(m)
                else:
                    other.append(m)
            #event_markers = [markers.pop(markers.index(m)) for m in markers if isinstance(m, EventMarker)]
            event_markers = sorted(event_markers,
                                   key=operator.attrgetter('_event.magnitude'), 
                                   reverse=order==Qt.DescendingOrder)

            event_markers.extend(other)
            self.viewer.markers = event_markers
        else:
            self.viewer.markers = sorted(self.viewer.markers, key=operator.attrgetter(_attr_mapping[col]))
            if order == Qt.DescendingOrder:
                self.viewer.markers.reverse()
        self.emit(SIGNAL("layoutChanged()"))

    def data(self, index, role):
        if not self.viewer:
            return QVariant()
        if role == Qt.DisplayRole:
            imarker = index.row()
            marker = self.viewer.markers[imarker]
            if index.column() == _column_mapping['Type']:
                if isinstance(marker, EventMarker):
                    s = 'E'
                elif isinstance(marker, PhaseMarker):
                    s = 'P'
                else:
                    s = ''

            if index.column() == _column_mapping['Time']:
                s = util.time_to_str(marker.tmin)


            if index.column() == _column_mapping['Magnitude']:
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
            if index.column() == 2 and isinstance(marker, EventMarker):
                marker.get_event().magnitude = value.toFloat()[0]
                self.emit(SIGNAL('dataChanged()'))
            return True
        return False

    def flags(self, index):
        if index.column() == _column_mapping['Magnitude'] and isinstance(self.viewer.markers[index.row()], EventMarker):
            return Qt.ItemFlags(35)
        return Qt.ItemFlags(33)


class MarkerEditor(QTableWidget):
    def __init__(self, *args, **kwargs):
        QTableWidget.__init__(self, *args, **kwargs)

        layout = QGridLayout()
        self.setLayout(layout)
        self.marker_table = MarkerTable()
        self.marker_table.setItemDelegate(MarkerItemDelegate(self.marker_table))

        self.marker_model = MarkerModel()
        delegate = MarkerItemDelegate()

        header = self.marker_table.horizontalHeader()
        header.setModel(self.marker_model)
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
        self.marker_table.set_viewer(self.viewer)

    def set_selected_markers(self, selected, deselected):
        ''' set markers selected in viewer at selection in table.'''
        selected_markers = [self.viewer.markers[i.row()] for i in self.selection_model.selectedRows()]
        self.viewer.set_selected_markers(selected_markers)

    def get_marker_model(self):
        '''Return MarkerModel instance'''
        return self.marker_model

    def update_selection_model(self, indices):
        ''' :param indices: list of indices of selected markers.'''
        self.selection_model.clearSelection()
        num_columns = len(_header_data)
        flag = QItemSelectionModel.SelectionFlags(2)
        selections = QItemSelection()
        for i in indices:
            left = self.marker_model.index(i, 0)
            right = self.marker_model.index(i, num_columns-1)
            row_selection = QItemSelection(left, right)
            row_selection.select(left, right)
            selections.merge(row_selection, flag)
        self.selection_model.select(selections, flag)


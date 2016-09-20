import numpy as num
from PyQt4 import QtCore as qc
from PyQt4 import QtGui as qg

from pyrocko.gui_util import EventMarker, PhaseMarker
from pyrocko import util, orthodrome

_header_data = [
    'T', 'Time', 'M', 'Label', 'Depth [km]', 'Lat', 'Lon', 'Kind', 'Dist [km]']

_column_mapping = dict(zip(_header_data, range(len(_header_data))))

_string_header = (_column_mapping['Time'], _column_mapping['Label'])


class MarkerItemDelegate(qg.QStyledItemDelegate):
    '''Takes are of the table's style.'''

    def __init__(self, *args, **kwargs):
        qg.QStyledItemDelegate.__init__(self, *args, **kwargs)


class MarkerSortFilterProxyModel(qg.QSortFilterProxyModel):
    '''Sorts the table's columns.'''

    def __init__(self):
        qg.QSortFilterProxyModel.__init__(self)
        self.sort(1, qc.Qt.AscendingOrder)

    def lessThan(self, left, right):
        if left.column() not in _string_header:
            return left.data().toDouble()[0] > right.data().toDouble()[0]
        else:
            return left > right


class MarkerTableView(qg.QTableView):
    def __init__(self, *args, **kwargs):
        qg.QTableView.__init__(self, *args, **kwargs)

        self.setSelectionBehavior(qg.QAbstractItemView.SelectRows)
        self.setSelectionMode(qg.QAbstractItemView.ContiguousSelection)
        self.setHorizontalScrollMode(qg.QAbstractItemView.ScrollPerPixel)
        self.setEditTriggers(qg.QAbstractItemView.DoubleClicked)
        self.setSortingEnabled(True)
        self.sortByColumn(1, qc.Qt.AscendingOrder)
        self.setAlternatingRowColors(True)

        self.setShowGrid(False)
        self.verticalHeader().hide()
        self.pile_viewer = None

        self.connect(self, qc.SIGNAL('clicked(QModelIndex)'), self.clicked)
        self.connect(
            self,
            qc.SIGNAL('doubleClicked(QModelIndex)'),
            self.double_clicked)

        self.header_menu = qg.QMenu(self)

        show_initially = ['Type', 'Time', 'Magnitude']
        self.menu_labels = ['Type', 'Time', 'Magnitude', 'Label', 'Depth [km]',
                            'Latitude/Longitude', 'Kind', 'Distance [km]']
        self.menu_items = dict(zip(self.menu_labels, [0, 1, 2, 3, 4, 5, 7, 8]))
        self.editable_columns_events = [2, 3, 4, 5, 6, 7]
        self.editable_columns_phases = [2, 3, 4, 5, 6, 7]
        self.editable_columns = \
            self.editable_columns_events + self.editable_columns_phases

        self.column_actions = {}
        for hd in self.menu_labels:
            a = qg.QAction(qc.QString(hd), self.header_menu)
            self.connect(a, qc.SIGNAL('triggered(bool)'), self.toggle_columns)
            a.setCheckable(True)
            if hd in show_initially:
                a.setChecked(True)
            else:
                a.setChecked(False)
            self.header_menu.addAction(a)
            self.column_actions[hd] = a

        header = self.horizontalHeader()
        header.setContextMenuPolicy(qc.Qt.CustomContextMenu)
        self.connect(
            header,
            qc.SIGNAL('customContextMenuRequested(QPoint)'),
            self.show_context_menu)

    def set_viewer(self, viewer):
        '''Set a pile_viewer and connect to signals.'''

        self.pile_viewer = viewer

    def keyPressEvent(self, key_event):
        self.pile_viewer.keyPressEvent(key_event)

    def clicked(self, model_index):
        '''Ignore mouse clicks.'''
        pass

    def double_clicked(self, model_index):
        if model_index.column() in self.editable_columns:
            return
        else:
            self.pile_viewer.go_to_selection()

    def show_context_menu(self, point):
        '''Pop-up menu to toggle columns in the :py:class:`MarkerTableView`.'''

        self.header_menu.popup(self.mapToGlobal(point))

    def toggle_columns(self):
        for header, ca in self.column_actions.items():
            hide = not ca.isChecked()
            if header == 'Latitude/Longitude':
                self.setColumnHidden(self.menu_items[header], hide)
                self.setColumnHidden(self.menu_items[header]+1, hide)
            else:
                self.setColumnHidden(self.menu_items[header], hide)
                if header == 'Dist [km]':
                    self.model().update_distances()


class MarkerTableModel(qc.QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        qc.QAbstractTableModel.__init__(self, *args, **kwargs)
        self.pile_viewer = None
        self.headerdata = _header_data
        self.distances = {}
        self.last_active_event = None
        self.row_count = 0

    def set_viewer(self, viewer):
        '''Set a pile_viewer and connect to signals.'''

        self.pile_viewer = viewer
        self.connect(self.pile_viewer,
                     qc.SIGNAL('markers_added(int,int)'),
                     self.markers_added)

        self.connect(self.pile_viewer,
                     qc.SIGNAL('markers_removed(int, int)'),
                     self.markers_removed)

        self.connect(self.pile_viewer,
                     qc.SIGNAL('changed_marker_selection'),
                     self.update_distances)

    def rowCount(self, parent):
        if not self.pile_viewer:
            return 0
        return len(self.pile_viewer.get_markers())

    def columnCount(self, parent):
        return len(_column_mapping)

    def markers_added(self, istart, istop):
        '''Insert rows into table.'''

        self.beginInsertRows(qc.QModelIndex(), istart, istop)
        self.endInsertRows()

    def markers_removed(self, i_from, i_to):
        '''Remove rows from table.'''

        self.beginRemoveRows(qc.QModelIndex(), i_from, i_to)
        self.endRemoveRows()
        self.marker_table_view.updateGeometries()

    def headerData(self, col, orientation, role):
        '''Set and format header data.'''

        if orientation == qc.Qt.Horizontal:
            if role == qc.Qt.DisplayRole:
                return qc.QVariant(self.headerdata[col])
            elif role == qc.Qt.SizeHintRole:
                return qc.QSize(10, 20)
        else:
            return qc.QVariant()

    def data(self, index, role):
        '''Set data in each of the table's cell.'''

        if not self.pile_viewer:
            return qc.QVariant()
        if role == qc.Qt.DisplayRole:
            imarker = index.row()
            marker = self.pile_viewer.markers[imarker]

            if index.column() == _column_mapping['T']:
                if isinstance(marker, EventMarker):
                    s = 'E'
                elif isinstance(marker, PhaseMarker):
                    s = 'P'
                else:
                    s = ''

            if index.column() == _column_mapping['Time']:
                s = util.time_to_str(marker.tmin)

            if index.column() == _column_mapping['M']:
                if isinstance(marker, EventMarker):
                    e = marker.get_event()
                    if e.moment_tensor is not None:
                        s = '%2.1f' % (e.moment_tensor.magnitude)
                    elif e.magnitude is not None:
                        s = '%2.1f' % (e.magnitude)
                    else:
                        s = ''
                else:
                    s = ''

            if index.column() == _column_mapping['Label']:
                if isinstance(marker, EventMarker):
                    s = str(marker.label())
                elif isinstance(marker, PhaseMarker):
                    s = str(marker.get_label())
                else:
                    s = ''

            if index.column() == _column_mapping['Depth [km]']:
                if isinstance(marker, EventMarker):
                    d = marker.get_event().depth
                    if d is not None:
                        s = '{0:4.1f}'.format(marker.get_event().depth/1000.)
                    else:
                        s = ''
                else:
                    s = ''

            if index.column() == _column_mapping['Lat']:
                if isinstance(marker, EventMarker):
                    s = '{0:4.2f}'.format(marker.get_event().lat)
                else:
                    s = ''

            if index.column() == _column_mapping['Lon']:
                if isinstance(marker, EventMarker):
                    s = '{0:4.2f}'.format(marker.get_event().lon)
                else:
                    s = ''

            if index.column() == _column_mapping['Kind']:
                s = '{:d}'.format(marker.kind)

            if index.column() == _column_mapping['Dist [km]']:
                if marker in self.distances.keys():
                    s = '{0:6.1f}'.format(self.distances[marker])
                else:
                    s = ''

            return qc.QVariant(qc.QString(s))

        return qc.QVariant()

    def update_distances(self, indices):
        '''Calculate and update distances between events.'''

        if len(indices) != 1 or self.marker_table_view.horizontalHeader()\
                .isSectionHidden(_column_mapping['Dist [km]']):
            return

        if self.last_active_event == self.pile_viewer.get_active_event():
            return
        else:
            self.last_active_event = self.pile_viewer.get_active_event()

        markers = self.pile_viewer.markers
        omarker = markers[indices[0]]
        if not isinstance(omarker, EventMarker):
            return

        emarkers = [m for m in markers if isinstance(m, EventMarker)]
        if len(emarkers) < 2:
            return

        lats = num.zeros(len(emarkers))
        lons = num.zeros(len(emarkers))
        for i in xrange(len(emarkers)):
            lats[i] = emarkers[i].get_event().lat
            lons[i] = emarkers[i].get_event().lon

        olats = num.zeros(len(emarkers))
        olons = num.zeros(len(emarkers))
        olats[:] = omarker.get_event().lat
        olons[:] = omarker.get_event().lon
        dists = orthodrome.distance_accurate50m_numpy(lats, lons, olats, olons)
        dists /= 1000.
        self.distances = dict(zip(emarkers, dists))
        self.marker_table_view.viewport().repaint()
        self.emit(qc.SIGNAL('dataChanged()'))

    def done(self):
        self.emit(qc.SIGNAL('dataChanged()'))
        return True

    def setData(self, index, value, role):
        '''Manipulate :py:class:`EventMarker` instances.'''

        if role == qc.Qt.EditRole:
            imarker = index.row()
            marker = self.pile_viewer.markers[imarker]
            if index.column() in [_column_mapping[c] for c in [
                    'M', 'Lat', 'Lon', 'Depth [km]']]:

                if not isinstance(marker, EventMarker):
                    return False
                else:
                    if index.column() == _column_mapping['M']:
                        valuef, valid = value.toFloat()
                        if valid:
                            e = marker.get_event()
                            if e.moment_tensor is None:
                                e.magnitude = valuef
                            else:
                                e.moment_tensor.magnitude = valuef
                            return self.done()

                if index.column() in [_column_mapping['Lon'],
                                      _column_mapping['Lat'],
                                      _column_mapping['Depth [km]']]:
                    if isinstance(marker, EventMarker):
                        valuef, valid = value.toFloat()
                        if valid:
                            if index.column() == _column_mapping['Lat']:
                                marker.get_event().lat = valuef
                            elif index.column() == _column_mapping['Lon']:
                                marker.get_event().lon = valuef
                            elif index.column() == _column_mapping[
                                    'Depth [km]']:
                                marker.get_event().depth = valuef*1000.
                            return self.done()

            if index.column() == _column_mapping['Label']:
                values = str(value.toString())
                if values != '':
                    if isinstance(marker, EventMarker):
                        marker.get_event().set_name(values)
                        return self.done()

                    if isinstance(marker, PhaseMarker):
                        marker.set_phasename(values)
                        return self.done()

        return False

    def flags(self, index):
        '''Set flags for cells which the user can edit.'''

        if index.column() not in self.marker_table_view.editable_columns:
            return qc.Qt.ItemFlags(33)
        else:
            if isinstance(self.pile_viewer.markers[index.row()], EventMarker):
                if index.column() in self.marker_table_view.editable_columns:
                    return qc.Qt.ItemFlags(35)
            if index.column() == _column_mapping['Label']:
                return qc.Qt.ItemFlags(35)
        return qc.Qt.ItemFlags(33)


class MarkerEditor(qg.QFrame):
    def __init__(self, *args, **kwargs):
        qg.QFrame.__init__(self, *args, **kwargs)

        layout = qg.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.marker_table = MarkerTableView()
        self.marker_table.setItemDelegate(
            MarkerItemDelegate(self.marker_table))

        self.marker_model = MarkerTableModel()
        self.marker_model.marker_table_view = self.marker_table

        self.proxy_filter = MarkerSortFilterProxyModel()
        self.proxy_filter.setDynamicSortFilter(True)
        self.proxy_filter.setSourceModel(self.marker_model)

        self.marker_table.setModel(self.proxy_filter)

        header = self.marker_table.horizontalHeader()
        header.setDefaultSectionSize(30)
        header.setResizeMode(0, qg.QHeaderView.Interactive)
        header.resizeSection(0, 40)
        for i in xrange(len(_header_data)):
            header.setResizeMode(i+2, qg.QHeaderView.Interactive)
            header.resizeSection(i+2, 70)
        header.setResizeMode(1, qg.QHeaderView.Interactive)
        header.resizeSection(1, 190)
        header.setStretchLastSection(True)

        self.setMinimumWidth(335)

        self.selection_model = qg.QItemSelectionModel(self.proxy_filter)
        self.marker_table.setSelectionModel(self.selection_model)
        self.connect(
            self.selection_model,
            qc.SIGNAL('selectionChanged(QItemSelection,QItemSelection)'),
            self.set_selected_markers)

        layout.addWidget(self.marker_table, 0, 0)

        self.pile_viewer = None

    def set_viewer(self, viewer):
        '''Set a pile_viewer and connect to signals.'''

        self.pile_viewer = viewer
        self.marker_model.set_viewer(viewer)
        self.marker_table.set_viewer(viewer)
        self.connect(
            self.pile_viewer,
            qc.SIGNAL('changed_marker_selection'),
            self.update_selection_model)

        self.connect(
            self.pile_viewer,
            qc.SIGNAL('markers_changed'),
            self.marker_table.viewport().repaint)

        self.marker_table.toggle_columns()

    def set_selected_markers(self, selected, deselected):
        ''' set markers selected in viewer at selection in table.'''

        selected_markers = []
        for i in self.selection_model.selectedRows():
            selected_markers.append(
                self.pile_viewer.markers[
                    self.proxy_filter.mapToSource(i).row()])

        self.pile_viewer.set_selected_markers(selected_markers)

    def get_marker_model(self):
        '''Return :py:class:`MarkerTableModel` instance'''

        return self.marker_model

    def update_selection_model(self, indices):
        '''Adopt marker selections done in the pile_viewer in the tableview.

        :param indices: list of indices of selected markers.'''

        self.selection_model.clearSelection()
        selections = qg.QItemSelection()
        num_columns = len(_header_data)
        flag = qg.QItemSelectionModel.SelectionFlags(
            (qg.QItemSelectionModel.Current | qg.QItemSelectionModel.Select))

        for i in indices:
            left = self.proxy_filter.mapFromSource(
                self.marker_model.index(i, 0))

            right = self.proxy_filter.mapFromSource(
                self.marker_model.index(i, num_columns-1))

            row_selection = qg.QItemSelection(left, right)
            row_selection.select(left, right)
            selections.merge(row_selection, flag)

        if len(indices) != 0:
            self.marker_table.setCurrentIndex(
                self.proxy_filter.mapFromSource(
                    self.marker_model.index(indices[0], 0)))
            self.selection_model.setCurrentIndex(
                self.proxy_filter.mapFromSource(
                    self.marker_model.index(indices[0], 0)),
                qg.QItemSelectionModel.SelectCurrent)

        self.selection_model.select(selections, flag)

        if len(indices) != 0:
            self.marker_table.scrollTo(
                self.proxy_filter.mapFromSource(
                    self.marker_model.index(indices[0], 0)))

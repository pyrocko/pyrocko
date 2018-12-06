
from pyrocko.gui.qt_compat import qg, qw


def get_err_palette():
    err_palette = qg.QPalette()
    err_palette.setColor(qg.QPalette.Text, qg.QColor(255, 200, 200))
    return err_palette


def get_palette():
    return qw.QApplication.palette()


def errorize(widget):
    widget.setStyleSheet('''
        QLineEdit {
            background: rgb(100, 20, 20);
        }''')


def de_errorize(widget):
    widget.setStyleSheet('')


def string_choices_to_combobox(cls):
    cb = qw.QComboBox()
    for i, s in enumerate(cls.choices):
        cb.insertItem(i, s)

    return cb

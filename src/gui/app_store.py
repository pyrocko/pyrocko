import sys
import json
try:
    from urllib2 import HTTPError, urlopen, URLError, Request
except ImportError:
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError

import os
import shutil
import logging
import re
import codecs

import PyQt5.QtGui as qg
import PyQt5.QtCore as qc
import PyQt5.QtWidgets as qw

from pyrocko import config as pconfig
from pyrocko.gui.util import WebKitFrame


logger = logging.getLogger('pyrocko.gui.app_store')

pjoin = os.path.join

git_branch = 'python3-appstore'
base_url = 'https://api.github.com/repos/HerrMuellerluedenscheid/contrib-snufflings/'
# base_url = 'https://api.github.com/repos/pyrocko/contrib-snufflings/'
exclude = ['.gitignore', 'screenshots', 'LICENSE', 'README.md']

destination_path = pconfig.config().snufflings


prolog_html = '''
    <html>
    <body>
    <h3>
    Add snufflings to your snuffler
    </h3>
    <p>
    The list of snufflings below mirrors the
    <a href="https://github.com/pyrocko/contrib-snufflings">user contributed snufflings
    repository</a><br>
    Checkout that site for bug reports, whishes and ideas.
    </p>
    <p>
    Note, that we trust authors of contrib snufflings and review most of the
    codes which are listed here.<br>
    However, we do not take any responsibilities.
    </p>
    </body>
    </html>
'''


class AppTile(qw.QWidget):

    want_install_snuffling = qc.pyqtSignal(str, dict)
    want_remove_snuffling = qc.pyqtSignal(str)
    open_url_signal = qc.pyqtSignal(str)

    def __init__(self, data, label, installed=False):
        qw.QWidget.__init__(self)
        self.data = data
        self.label = label
        self.setLayout(qw.QHBoxLayout())
        self.setup()
        self.update_state(installed)
        self.snuffling = None

    def setup(self):
        self.layout().addWidget(qw.QLabel(self.label))
        self.button_install = qw.QPushButton('')
        self.button_install.clicked.connect(self.install_remove)
        self.layout().addWidget(self.button_install)

        button_checkout = qw.QPushButton('Source Code')
        self.layout().addWidget(button_checkout)
        button_checkout.clicked.connect(
            self.open_url_slot)

    @qc.pyqtSlot()
    def open_url_slot(self):
        self.open_url_signal.emit(self.data['html_url'])

    def update_state(self, is_installed=None):
        if is_installed or self.is_installed:
            button_label = 'Remove'
        else:
            button_label = 'Install'

        self.button_install.setText(button_label)

    @qc.pyqtSlot()
    def install_remove(self):
        if not self.is_installed:
            self.want_install_snuffling.emit(self.data['name'], self.data)
        else:
            self.want_remove_snuffling.emit(self.data['name'])

    @property
    def is_installed(self):
        installed_snufflings = os.listdir(destination_path)
        return self.data['name'] in installed_snufflings

    def connect_to_installer(self, install_service):
        self.want_install_snuffling.connect(install_service.install)
        self.want_remove_snuffling.connect(install_service.remove)


class AppWidget(qw.QWidget):
    def __init__(self, viewer=None, install_service=None, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.url = base_url + 'contents' + '?ref=%s' % git_branch
        self.json_data = None
        self.install_service = install_service
        self.viewer = viewer
        self.setLayout(qw.QVBoxLayout())
        self.tiles = []

    def fail(self, message):
        box = qw.QMessageBox(self.viewer)
        logger.debug(message)
        box.setText('%s' % message)
        box.exec_()

    def refresh(self, callback=None):
        try:
            logger.info('Checking contrib-snufflings repo')

            nbytes_header = 100
            header_request = ('Range', 'bytes=0-%s' % nbytes_header)
            reader_re = re.compile('name\s*=(.*)', flags=re.IGNORECASE)
            response = urlopen(self.url)
            reader = codecs.getreader("utf-8")
            self.json_data = json.load(reader(response))

            n_repos = len(self.json_data)
            progress_dialog = self.get_progressbar()
            progress_dialog.setMaximum(n_repos)
            progress_dialog.show()
            self.refresh_cancelled = False

            for irepo, data in enumerate(self.json_data):
                progress_dialog.setValue(irepo+1)

                if self.refresh_cancelled:
                    return

                if data['name'] in exclude:
                    continue

                if data['type'] == 'dir':
                    req = Request(
                        base_url + 'contents/' + data['name'] +'/snuffling.py' + '?ref=%s'%git_branch)
                else:
                    req = Request(data['download_url'])
                
                req.add_header(*header_request)
                response = urlopen(req)

                app_header = reader(response).read()

                m = re.search(reader_re, app_header)
                print(app_header)
                if m:
                    app_label = m.group(1).strip()
                else:
                    app_label = data['name']

                tile = AppTile(
                    data,
                    label=app_label,
                    installed=self.install_service.is_installed(data['name']))

                tile.connect_to_installer(self.install_service)
                tile.open_url_signal.connect(self.open_browser)

                self.layout().addWidget(tile)

        except URLError as e:
            self.fail('No connection to internet')
            print(e)
            self.setVisible(False)
        finally:
            progress_dialog.setValue(n_repos)

    def get_progressbar(self):
        progress_dialog = qw.QProgressDialog(parent=self)
        progress_dialog.setModal(qc.Qt.WindowModal)
        progress_dialog.setWindowTitle('Connecting')
        progress_dialog.setLabelText('Connecting to AppStore')
        progress_dialog.setAutoClose(True)
        progress_dialog.canceled.connect(self.cancel_refresh)
        return progress_dialog

    def cancel_refresh(self):
        self.refresh_cancelled = True

    @qc.pyqtSlot(str)
    def open_browser(self, url):
        f = WebKitFrame(url)
        if self.viewer:
            self.viewer.panel_parent.add_tab('Browse the code', f)
        else:
            logger.warn('No viewer available')


class Installer:
    ''' 
    Downloads, installs and removes snuffling modules from the *pile_viewer*
    '''
    def __init__(self, pile_viewer):
        self.pile_viewer = pile_viewer
        self.snufflings = {}

    def download(self, data):
        name = data['name']
        logger.debug('Downloading snuffling %s' % name)
        if data['type'] == 'dir':
            response = urlopen(
                base_url + 'contents/' + name + '?ref=%s'%git_branch)
            logger.debug(response)
            json_data = json.load(response.decode())

            os.mkdir(pjoin(destination_path, name))
            for item in json_data:
                self.download(name, item)
        else:
            response = urlopen(data['download_url'])
            with open(pjoin(destination_path, name), 'w') as f:
                fd = response.read()
                f.write(fd)

    def is_installed(self, name):
        for (directory, sname) in self.snufflings.keys():
            if sname == name:
                return True
        return False

    @qc.pyqtSlot(str, str)
    def install(self, name, data):
        self.download(data)
        directory = pjoin(destination_path, name)
        snuffling = pyrocko.gui.snuffling.SnufflingModule(
            directory, name, self.pile_viewer)
        
        self.snuffling_modules[directory, name] = suffling
        self.pile_viewer.add_snuffling(snuffling)

    @qc.pyqtSlot(str)
    def remove(self, name):
        logger.debug('Delete snuffling %s' % name)
        directory = pjoin(destination_path, name)
        self.pile_viewer.remove_snuffling(self.snuffling_modulesp[directory, name])
        try:
            shutil.rmtree(directory)
        except OSError as e:
            os.remove(fns)
            print(e)

    def setup_snufflings(self):
        if self.viewer:
            logger.debug('setup snufflings')
            self.viewer.setup_snufflings([destination_path])


class AppStore(qw.QFrame):
    def __init__(self, viewer=None, *args, **kwargs):
        qw.QFrame.__init__(self, *args, **kwargs)
        self.viewer = viewer
        install_service = Installer(pile_viewer=self.viewer)
        w = AppWidget(viewer=self.viewer, install_service=install_service)
        w.refresh()

        self.setLayout(qw.QVBoxLayout())
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(qg.QPalette.Background, qc.Qt.white)
        self.setPalette(pal)

        prolog = qw.QLabel(prolog_html)

        scroller = qw.QScrollArea(parent=self)
        scroller.setWidget(w)
        w.setParent(scroller)

        self.layout().addWidget(prolog)
        self.layout().addWidget(scroller)

    def closeEvent(self, event):
        # TODO connect to closing event in pile viewer
        if self.viewer:
            self.viewer.store = None



if __name__ == '__main__':
    if len(sys.argv) > 2 and '--debug' in sys.argv[2]:
        logger.setLevel(logging.DEBUG)
    app = qw.QApplication(sys.argv)
    win = qw.QMainWindow()
    w = AppStore()
    win.setCentralWidget(w)
    win.show()
    sys.exit(app.exec_())

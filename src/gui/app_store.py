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

    snuffling_update_required = qc.pyqtSignal()
    open_url_signal = qc.pyqtSignal(str)

    def __init__(self, data, label, installed=False):
        qw.QWidget.__init__(self)
        self.data = data
        self.label = label
        self.installed = installed
        self.setLayout(qw.QHBoxLayout())
        self.setup()
        self.update_state(installed)

    def setup(self):
        self.layout().addWidget(qw.QLabel(self.label))
        self.button_install = qw.QPushButton('')
        self.button_install.clicked.connect(self.un_install)
        self.layout().addWidget(self.button_install)

        button_checkout = qw.QPushButton('View')
        self.layout().addWidget(button_checkout)
        print(self.data['url'])
        button_checkout.clicked.connect(
            self.open_url_slot)

    @qc.pyqtSlot()
    def open_url_slot(self):
        self.open_url.emit(self.data['url'])

    def update_state(self, is_installed=None):
        if is_installed or self.is_installed:
            button_label = 'Remove'
        else:
            button_label = 'Install'

        self.button_install.setText(button_label)
        self.snuffling_update_required.emit()

    def download(self, name, item):
        logger.debug('Downloading snuffling %s' % item['name'])
        response = urlopen(item['download_url'])
        with open(pjoin(destination_path, name, item['name']), 'w') as f:
            fd = response.read()
            f.write(fd)

    @property
    def is_installed(self):
        installed_snufflings = os.listdir(destination_path)
        return self.data['name'] in installed_snufflings

    def un_install(self):
        if self.is_installed:
            self.remove()
        else:
            if self.data['type'] == 'dir':
                response = urlopen(base_url + 'contents/' + self.data['name'] + '?ref=%s'%git_branch)
                logger.debug(response)
                json_data = json.load(response.decode())

                name = self.data['name']
                os.mkdir(pjoin(destination_path, name))
                for item in json_data:
                    self.download(name, item)
            else:
                self.download('', self.data)

        self.update_state()

    def remove(self):
        logger.debug('Delete snuffling %s' % self.data['name'])
        fns = pjoin(destination_path, self.data['name'])
        try:
            shutil.rmtree(fns)
        except OSError:
            os.remove(fns)


class AppWidget(qw.QWidget):
    def __init__(self, viewer=None, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.url = base_url + 'contents' + '?ref=%s' % git_branch
        self.json_data = None
        self.viewer = viewer
        self.setLayout(qw.QVBoxLayout())

    def fail(self, message):
        box = qw.QMessageBox(self.viewer)
        logger.debug(message)
        box.setText('%s' % message)
        box.exec_()

    def refresh(self):
        try:
            logger.info('Checking contrib-snufflings repo')

            nbytes_header = 100
            header_request = ('Range', 'bytes=0-%s' % nbytes_header)
            reader_re = re.compile('name\s*=(.*)', flags=re.IGNORECASE)
            
            response = urlopen(self.url)
            reader = codecs.getreader("utf-8")
            self.json_data = json.load(reader(response))

            n_repos = len(self.json_data)
            layout = self.layout()
            for irepo, data in enumerate(self.json_data):
                print(irepo+1, n_repos)
                if data['name'] in exclude:
                    continue

                try:
                    installed_snufflings = os.listdir(destination_path)
                    is_installed = data['name'] in installed_snufflings
                except OSError as e:
                    if e.errno == 2:
                        os.mkdir(destination_path)
                        is_installed = False
                    else:
                        raise e

                if data['type'] == 'dir':
                    # TODO: to be handled
                    continue

                req = Request(data['download_url'])
                req.add_header(*header_request)
                response = urlopen(req)

                app_header = reader(response).read()

                m = re.search(reader_re, app_header)
                if m:
                    app_label = m.group(1)
                else:
                    app_label = data['name']
                tile = AppTile(data, label=app_label, installed=is_installed)
                tile.open_url_signal.connect(self.parent().open_browser)
                tile.snuffling_update_required.connect(self.setup_snufflings)
                self.layout().addWidget(tile)

        except URLError as e:
            self.fail('No connection to internet')
            self.setVisible(False)

    def setup_snufflings(self):
        if self.viewer:
            logger.debug('setup snufflings')
            self.viewer.setup_snufflings([destination_path])


class AppStore(qw.QFrame):
    def __init__(self, viewer=None, *args, **kwargs):
        qw.QFrame.__init__(self, *args, **kwargs)
        self.viewer = viewer
        w = AppWidget(viewer=self.viewer)
        w.refresh()
        self.setLayout(qw.QVBoxLayout())

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

    @qc.pyqtSlot(str)
    def open_browser(self, url):
        f = WebKitFrame(url)
        self.viewer.add_tab('Browse the code', f)


if __name__ == '__main__':
    if len(sys.argv) > 2 and '--debug' in sys.argv[2]:
        logger.setLevel(logging.DEBUG)
    app = qw.QApplication(sys.argv)
    win = qw.QMainWindow()
    w = AppStore()
    win.setCentralWidget(w)
    win.show()
    sys.exit(app.exec_())

import sys
import json
try:
    from urllib2 import HTTPError, urlopen, URLError
except ImportError:
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError

import os
import shutil
import logging

import PyQt4.QtGui as qg
import PyQt4.QtCore as qc
from pyrocko import config as pconfig

logger = logging.getLogger('pyrocko.gui.app_store')

pjoin = os.path.join

base_url = 'https://api.github.com/repos/pyrocko/contrib-snufflings/'
exclude = ['.gitignore', 'screenshots', 'LICENSE', 'README.md']

destination_path = pconfig.config().snufflings


class AppTile(qg.QWidget):
    def __init__(self, data, installed=False):
        qg.QWidget.__init__(self)
        self.data = data
        self.installed = installed
        self.setLayout(qg.QHBoxLayout())
        self.setup()
        self.update_state(installed)

    def setup(self):
        name = self.data['name']
        self.layout().addWidget(qg.QLabel(name))
        self.install_button = qg.QPushButton('')
        self.install_button.clicked.connect(self.un_install)
        self.layout().addWidget(self.install_button)

    def update_state(self, is_installed=None):
        if is_installed or self.is_installed:
            button_label = 'remove'
        else:
            button_label = 'install'

        self.install_button.setText(button_label)
        self.emit(qc.SIGNAL('snuffling_update_required()'))

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
                response = urlopen(base_url + 'contents/' + self.data['name'])
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


class AppWidget(qg.QWidget):
    def __init__(self, viewer=None, *args, **kwargs):
        qg.QWidget.__init__(self, *args, **kwargs)
        self.url = base_url + 'contents'
        self.json_data = None
        self.viewer = viewer
        self.setLayout(qg.QVBoxLayout())

    def fail(self, message):
        box = qg.QMessageBox(self.viewer)
        self.debug(message)
        box.setText('%s' % message)
        box.exec_()

    def refresh(self):
        try:
            logger.debug('Checking contrib-snufflings repo')
            response = urlopen(self.url)
            self.json_data = json.load(response)
            layout = self.layout()

            for data in self.json_data:
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

                tile = AppTile(data, installed=is_installed)
                self.connect(
                    tile,
                    qc.SIGNAL('snuffling_update_required()'),
                    self.setup_snufflings)
                layout.addWidget(tile)

        except URLError:
            self.fail('No connection to internet')

    def setup_snufflings(self):
        if self.viewer:
            logger.debug('setup snufflings')
            self.viewer.setup_snufflings([destination_path])


class AppStore(qg.QFrame):
    def __init__(self, viewer=None, *args, **kwargs):
        qg.QFrame.__init__(self, *args, **kwargs)
        w = AppWidget(viewer=viewer)
        w.refresh()
        self.setLayout(qg.QVBoxLayout())
        scroller = qg.QScrollArea()
        scroller.setWidget(w)
        self.layout().addWidget(scroller)


if __name__ == '__main__':
   app = qg.QApplication(sys.argv)
   w = AppStore()
   w.show()
   sys.exit(app.exec_())

import sys
import json
import urllib2
import os
import shutil

import PyQt4.QtGui as qg
import PyQt4.QtCore as qc
from pyrocko import config as pconfig

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

    def download(self, name, item):
        response = urllib2.urlopen(item['download_url'])
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
                response = urllib2.urlopen(base_url + 'contents/' + self.data['name'])
                json_data = json.load(response)

                # download files
                name = self.data['name']
                os.mkdir(pjoin(destination_path, name))
                for item in json_data:
                    self.download(name, item)

            else:
                self.download('', self.data)

        self.update_state()

    def remove(self):
        fns = pjoin(destination_path, self.data['name'])
        try:
            shutil.rmtree(fns)
        except OSError:
            os.remove(fns)


class AppWidget(qg.QWidget):
    def __init__(self):
        qg.QWidget.__init__(self)
        self.url = base_url + 'contents'
        self.json_data = None
        self.setLayout(qg.QVBoxLayout())

    def refresh(self):
        response = urllib2.urlopen(self.url)
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
            layout.addWidget(tile)


class AppStore(qg.QFrame):
    def __init__(self, *args, **kwargs):
        qg.QFrame.__init__(self, *args, **kwargs)
        w = AppWidget()
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

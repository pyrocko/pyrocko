from pyrocko.serial_hamster import Acquisition
from pyrocko import util
from subprocess import Popen, PIPE

util.setup_logging('test_hamster', 'debug')

source = Popen(['./test/datasource_deprecated.py'], stdout=PIPE)

seis = Acquisition(in_file=source.stdout)

seis.start()

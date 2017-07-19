from pyrocko.serial_hamster import SerialHamster
from pyrocko import util
from subprocess import Popen, PIPE

util.setup_logging('test_hamster', 'debug')

source = Popen(['./test_datasource.py'], stdout=PIPE)

seis = SerialHamster(in_file=source.stdout)

seis.start()

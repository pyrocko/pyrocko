import os
import pickle
import struct
import sys
import errno


class EOF(Exception):
    pass


class NoSuchCommand(Exception):
    pass


def readn(fh, size):
    nread = 0
    while nread < size:
        while True:
            try:
                data = os.read(fh, size-nread)
                break

            except OSError, e:
                if e.errno == errno.EINTR:
                    pass
                else:
                    raise e

        if len(data) == 0:
            raise EOF()
        nread += len(data)
    return data


def readobj(fh):
    data = readn(fh, 8)
    if len(data) == 0:
        raise EOF()
    size = struct.unpack('>Q', data)[0]
    data = readn(fh, size)
    payload = pickle.loads(data)
    return payload


def writeobj(fh, obj):
    data = pickle.dumps(obj)
    os.write(fh, struct.pack('>Q', len(data)))
    os.write(fh, data)


class Forked:
    def __init__(self, flipped=False):
        self.pid = None
        self.down_w = None
        self.up_r = None
        self.down_r = None
        self.up_w = None
        self.flipped = flipped
        self.commands = []

    def __getattr__(self, k):
        def f(*args, **kwargs):
            return self.call(k, *args, **kwargs)
        return f

    def dispatch(self, command, args, kwargs):
        if command in self.commands:
            return getattr(self, command+'_')(*args, **kwargs)
        else:
            raise NoSuchCommand(command)

    def start(self):
        down_r, down_w = os.pipe()
        up_r, up_w = os.pipe()
        self.pid = os.fork()

        if (not self.flipped and self.pid == 0) or (
                self.flipped and self.pid != 0):

            os.close(down_w)
            os.close(up_r)
            self.up_w = up_w
            self.down_r = down_r
            self.run()
            os.close(self.up_w)
            os.close(self.down_r)
            self.up_w = None
            self.down_r = None
            if self.pid == 0:
                sys.exit()
            else:
                os.wait()
        else:
            os.close(up_w)
            os.close(down_r)
            self.down_w = down_w
            self.up_r = up_r

    def run(self):
        while self.process():
            pass

    def process(self):
        try:
            payload = readobj(self.down_r)
        except EOF:
            return False
        exception = None
        payback = None
        try:
            payback = self.dispatch(*payload)
        except Exception, exception:
            pass
        writeobj(self.up_w, (payback, exception))
        return True

    def call(self, command, *args, **kwargs):
        if self.down_w is None:
            self.start()

        if self.down_w is not None:
            writeobj(self.down_w, (command, args, kwargs))
            payback, exception = readobj(self.up_r)
            if exception is not None:
                raise exception
            return payback
        else:
            sys.exit()

    def close(self):
        if self.down_w is not None:
            import os
            os.close(self.down_w)
            os.close(self.up_r)
            self.down_w = None
            self.up_r = None

        if self.pid is not None and self.pid != 0:
            os.wait()

    def __del__(self):
        if self.down_w is not None:
            self.close()

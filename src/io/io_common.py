# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------


class FileError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        self.context = {}

    def set_context(self, k, v):
        self.context[k] = v

    def __str__(self):
        s = Exception.__str__(self)
        if not self.context:
            return s

        lines = []
        for k in sorted(self.context.keys()):
            lines.append('%s: %s\n' % (k, self.context[k]))

        return '%s\n%s' % (s, '\n'.join(lines))


class FileLoadError(FileError):
    '''
    Raised when a problem occurred while loading of a file.
    '''


class FileSaveError(FileError):
    '''
    Raised when a problem occurred while saving of a file.
    '''

from pyrocko import guts
from pyrocko.guts import TBase, Object, ValidationError
import numpy as num
from cStringIO import StringIO
from base64 import b64decode


class literal(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar(
        'tag:yaml.org,2002:str', str(data), style='|')


guts.SafeDumper.add_representer(literal, literal_presenter)


class Array(Object):

    dummy_for = num.ndarray

    class __T(TBase):
        def __init__(
                self,
                shape=None,
                dtype=None,
                serialize_as='table',
                serialize_dtype=None,
                *args, **kwargs):

            TBase.__init__(self, *args, **kwargs)
            self.shape = shape
            self.dtype = dtype
            assert serialize_as in ('table', 'base64', 'list')
            self.serialize_as = serialize_as
            self.serialize_dtype = serialize_dtype

        def regularize_extra(self, val):
            if isinstance(val, basestring):
                ndim = None
                if self.shape:
                    ndim = len(self.shape)

                if self.serialize_as == 'table':
                    val = num.loadtxt(
                        StringIO(str(val)), dtype=self.dtype, ndmin=ndim)

                elif self.serialize_as == 'base64':
                    data = b64decode(val)
                    val = num.fromstring(
                        data, dtype=self.serialize_dtype).astype(self.dtype)
            else:
                val = num.asarray(val, dtype=self.dtype)

            return val

        def validate_extra(self, val):
            if self.dtype != val.dtype:
                raise ValidationError(
                    'array not of required type: need %s, got %s' % (
                        self.dtype, val.dtype))

            if self.shape is not None:
                la, lb = len(self.shape), len(val.shape)
                if la != lb:
                    raise ValidationError(
                        'array dimension mismatch: need %i, got %i' % (
                            la, lb))

                for a, b in zip(self.shape, val.shape):
                    if a is not None:
                        if a != b:
                            raise ValidationError(
                                'array shape mismatch: need %s, got: %s' % (
                                    self.shape, val.shape))

        def to_save(self, val):
            if self.serialize_as == 'table':
                out = StringIO()
                num.savetxt(out, val, fmt='%12.7g')
                return literal(out.getvalue())
            elif self.serialize_as == 'base64':
                data = val.astype(self.serialize_dtype).tostring()
                return literal(data.encode('base64'))
            elif self.serialize_as == 'list':
                if self.dtype == num.complex:
                    return [repr(x) for x in val]
                else:
                    return val.tolist()


__all__ = ['Array']

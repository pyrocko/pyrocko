
import unittest
import calendar
import math
import re
import sys
from contextlib import contextmanager


from pyrocko.guts import StringPattern, Object, Bool, Int, Float, String, \
    SObject, Unicode, Complex, Timestamp, DateTimestamp, StringChoice, Defer, \
    ArgumentError, ValidationError, Any, List, Tuple, Union, Choice, \
    load, load_string, load_xml_string, load_xml, load_all, iload_all, \
    load_all_xml, iload_all_xml, dump, dump_xml, dump_all, dump_all_xml


class SamplePat(StringPattern):
    pattern = r'[a-z]{3}'


class SampleChoice(StringChoice):
    choices = ['a', 'bcd', 'efg']


basic_types = (
    Bool, Int, Float, String, Unicode, Complex, Timestamp, SamplePat,
    SampleChoice)


def tstamp(*args):
    return float(calendar.timegm(args))


samples = {}
samples[Bool] = [True, False]
samples[Int] = [2**n for n in [1, 30]]  # ,31,65] ]
samples[Float] = [0., 1., math.pi, float('inf'), float('-inf'), float('nan')]
samples[String] = [
    '', 'test', 'abc def', '<', '\n', '"', '\'',
    ''.join(chr(x) for x in range(32, 128))]
# chr(0) and other special chars don't work with xml...

samples[Unicode] = [u'aoeu \u00e4 \u0100']
samples[Complex] = [1.0+5J, 0.0J, complex(math.pi, 1.0)]
samples[Timestamp] = [
    0.0,
    tstamp(2030, 1, 1, 0, 0, 0),
    tstamp(1960, 1, 1, 0, 0, 0),
    tstamp(2010, 10, 10, 10, 10, 10) + 0.000001]

samples[SamplePat] = ['aaa', 'zzz']
samples[SampleChoice] = ['a', 'bcd', 'efg']

regularize = {}
regularize[Bool] = [(1, True), (0, False), ('0', False), ('False', False)]
regularize[Int] = [('1', 1), (1.0, 1), (1.1, 1)]
regularize[Float] = [
    ('1.0', 1.0),
    (1, 1.0),
    ('inf', float('inf')),
    ('nan', float('nan'))]
regularize[String] = [(1, '1')]
regularize[Unicode] = [(1, u'1')]
regularize[Timestamp] = [
    ('2010-01-01 10:20:01',  tstamp(2010, 1, 1, 10, 20, 1)),
    ('2010-01-01T10:20:01',  tstamp(2010, 1, 1, 10, 20, 1)),
    ('2010-01-01T10:20:01.11Z',  tstamp(2010, 1, 1, 10, 20, 1)+0.11),
    ('2030-12-12 00:00:10.11111',  tstamp(2030, 12, 12, 0, 0, 10)+0.11111)
]


class GutsTestCase(unittest.TestCase):

    if sys.version_info < (2, 7):

        @contextmanager
        def assertRaises(self, exc):

            gotit = False
            try:
                yield None
            except exc:
                gotit = True

            assert gotit, 'expected to get a %s exception' % exc

        def assertIsNone(self, value):
            assert value is None, 'expected None but got %s' % value

    def assertEqualNanAware(self, a, b):
        if isinstance(a, float) \
                and isinstance(b, float) \
                and math.isnan(a) and math.isnan(b):
            return

        self.assertEqual(a, b)

    def testStringChoice(self):
        class X(Object):
            m = StringChoice.T(['a', 'b'])

        x = X(m='a')
        x.validate()
        x = X(m='c')
        with self.assertRaises(ValidationError):
            x.validate()

    def testAny(self):
        class X(Object):
            y = Int.T(default=1)

        class A(Object):
            x = Any.T()
            l = List.T(Any.T())

        a1 = A(x=X(y=33))
        a1.validate()
        a1c = load_string(a1.dump())
        self.assertEqual(a1c.x.y, 33)

        a2 = A(x=22, l=['a', 44, X(y=22)])
        a2.validate()
        a2c = load_string(a2.dump())
        self.assertEqual(a2c.x, 22)
        self.assertEqual(a2c.l[:2], ['a', 44])
        self.assertEqual(a2c.l[2].y, 22)

        a3 = A(x=X(y='10'))
        with self.assertRaises(ValidationError):
            a3.validate()

        a3.regularize()
        assert a3.x.y == 10

    def testAnyObject(self):
        class Base(Object):
            pass

        class A(Base):
            y = Int.T(default=1)

        class B(Base):
            x = Float.T(default=1.0)

        class C(Object):
            c = Base.T(optional=True)

        ca = C(c=A())
        cb = C(c=B())
        c_err = C(c='abc')
        cc_err = C(c=C())
        ca_err = C(c=A(y='1'))

        for c in (ca, cb):
            c.validate()

        for c in (c_err, cc_err, ca_err):
            with self.assertRaises(ValidationError):
                c.validate()

        ca_err.regularize()
        assert ca_err.c.y == 1

    def testUnion(self):

        class X1(Object):
            xmltagname = 'root'
            m = Union.T(members=[Int.T(), StringChoice.T(['small', 'large'])])

        class U(Union):
            members = [Int.T(), StringChoice.T(['small', 'large'])]

        class X2(Object):
            xmltagname = 'root'
            m = U.T()

        for X in [X1, X2]:
            X = X1
            x1 = X(m='1')
            with self.assertRaises(ValidationError):
                x1.validate()

            x1.validate(regularize=True)
            self.assertEqual(x1.m, 1)

            x2 = X(m='small')
            x2.validate()
            x3 = X(m='fail!')
            with self.assertRaises(ValidationError):
                x3.validate()
            with self.assertRaises(ValidationError):
                x3.validate(regularize=True)

            for x in [x1, x2]:
                y = load_string(x.dump())
                self.assertEqual(x.m, y.m)

            for x in [x1, x2]:
                y = load_xml_string(x.dump_xml())
                self.assertEqual(x.m, y.m)

    def testChoice(self):
        class A(Object):
            x = Int.T(default=1)

        class B(Object):
            y = Int.T(default=2)

        class OhOh(Object):
            z = Float.T(default=4.0)

        class AB(Choice):
            choices = [A.T(xmltagname='a'), B.T(xmltagname='b')]

        class C1(Object):
            xmltagname = 'root1'
            ab = Choice.T(choices=[A.T(xmltagname='a'), B.T(xmltagname='b')])

        class C2(Object):
            xmltagname = 'root2'
            ab = AB.T()

        for C in (C1, C2):
            for good in (A(), B()):
                c = C(ab=good)
                c.validate()
                for dumpx, loadx in (
                        (dump, load_string),
                        (dump_xml, load_xml_string)):
                    c2 = loadx(dumpx(c))
                    self.assertEqual(type(c2), C)
                    self.assertEqual(type(c2.ab), type(good))

            for bad in (OhOh(), 5.0):
                c = C(ab=bad)
                with self.assertRaises(ValidationError):
                    c.validate()

    def testTooMany(self):

        class A(Object):
            m = List.T(Int.T())
            xmltagname = 'a'

        a = A(m=[1, 2, 3])

        class A(Object):
            m = Int.T()
            xmltagname = 'a'

        with self.assertRaises(ArgumentError):
            load_xml_string(a.dump_xml())

    def testDeferred(self):
        class A(Object):
            p = Defer('B.T', optional=True)

        class B(Object):
            p = A.T()

        a = A(p=B(p=A()))
        del a

    def testListDeferred(self):
        class A(Object):
            p = List.T(Defer('B.T'))
            q = List.T(Defer('B.T'))

        class B(Object):
            p = List.T(A.T())

        a = A(p=[B(p=[A()])], q=[B(), B()])
        del a

    def testOptionalList(self):
        class A(Object):
            l1 = List.T(Int.T())
            l2 = List.T(Int.T(), optional=True)
            l3 = List.T(Int.T(), default=[1, 2, 3])

        a = A()
        a.validate()
        self.assertEqual(a.l1, [])
        self.assertIsNone(a.l2)
        self.assertEqual(a.l3, [1, 2, 3])

    def testSelfDeferred(self):
        class A(Object):
            a = Defer('A.T', optional=True)

        a = A(a=A(a=A()))
        del a

    def testOrder(self):
        class A(Object):
            a = Int.T(default=1)
            b = Int.T(default=2)

        class B(A):
            a = Int.T(default=3, position=A.a.position)
            c = Int.T(default=4)

        b = B()
        names = [k for (k, v) in b.T.inamevals(b)]
        assert names == ['a', 'b', 'c']

    def testOrderDeferred(self):
        class A(Object):
            a = Defer('B.T')
            b = Int.T(default=0)

        class B(Object):
            x = Int.T(default=1)

        a = A(a=B())
        names = [k for (k, v) in a.T.inamevals(a)]

        assert names == ['a', 'b']

    def testDocs(self):
        class A(Object):
            '''A description'''

            a = Int.T(default=0, help='the a property')

        class B(A):
            '''B description'''

            b = Float.T(default=0, help='the b property')
            c = List.T(Int.T(), help='the c')

        self.assertEqual(B.__doc__, '''B description

    .. py:gattribute:: b

      ``float``, *default:* ``0``
      
      the b property

    .. py:gattribute:: c

      ``list`` of ``int`` objects, *default:* ``[]``
      
      the c
''')  # noqa

    def testContentStyleXML(self):

        class Duration(Object):
            unit = String.T(optional=True, xmlstyle='attribute')
            uncertainty = Float.T(optional=True)
            value = Float.T(optional=True, xmlstyle='content')

            xmltagname = 'duration'

        s = '<duration unit="s"><uncertainty>1.0</uncertainty>10.5</duration>'
        dur = load_xml_string(s)
        self.assertEqual(dur.value, float('10.5'))
        self.assertEqual(dur.unit, 's')
        self.assertEqual(dur.uncertainty, float('1.0'))
        self.assertEqual(re.sub(r'\n\s*', '', dur.dump_xml()), s)

    def testSObject(self):

        class DotName(SObject):
            network = String.T()
            station = String.T()

            def __init__(self, s=None, **kwargs):
                if s is not None:
                    network, station = s.split('.')
                    kwargs = dict(network=network, station=station)

                SObject.__init__(self, **kwargs)

            def __str__(self):
                return '.'.join((self.network, self.station))

        class X(Object):
            xmltagname = 'root'
            dn = DotName.T()

        x = X(dn=DotName(network='abc', station='def'))
        self.assertEqual(x.dn.network, 'abc')
        self.assertEqual(x.dn.station, 'def')
        x = load_string(dump(x))
        self.assertEqual(x.dn.network, 'abc')
        self.assertEqual(x.dn.station, 'def')
        x = load_xml_string(dump_xml(x))
        self.assertEqual(x.dn.network, 'abc')
        self.assertEqual(x.dn.station, 'def')

    def testArray(self):
        from pyrocko.guts_array import Array
        import numpy as num

        shapes = [(None,), (1,), (10,), (1000,)]
        for shape in shapes:
            for serialize_as in ('base64', 'table', 'npy', 'base64+meta'):
                class A(Object):
                    xmltagname = 'aroot'
                    arr = Array.T(
                        shape=shape,
                        dtype=num.int,
                        serialize_as=serialize_as,
                        serialize_dtype='>i8')

                n = shape[0] or 10
                a = A(arr=num.arange(n, dtype=num.int))

                b = load_string(a.dump())
                self.assertTrue(num.all(a.arr == b.arr))

                if serialize_as is not 'base64+meta':
                    b = load_xml_string(a.dump_xml())
                    self.assertTrue(num.all(a.arr == b.arr))

                if shape[0] is not None:
                    with self.assertRaises(ValidationError):
                        a = A(arr=num.arange(n+10, dtype=num.int))
                        a.validate()

        for s0 in [None, 2, 10]:
            for s1 in [None, 2, 10]:
                class A(Object):
                    xmltagname = 'aroot'
                    arr = Array.T(
                        shape=(s0, s1),
                        dtype=num.int)

                n0 = s0 or 100
                n1 = s1 or 100
                a = A(arr=num.arange(n0*n1, dtype=num.int).reshape((n0, n1)))
                b = load_string(a.dump())
                self.assertTrue(num.all(a.arr == b.arr))

                b = load_xml_string(a.dump_xml())
                self.assertTrue(num.all(a.arr == b.arr))

    def testArrayNoDtype(self):
        from pyrocko.guts_array import Array
        import numpy as num

        class A(Object):
            arr = Array.T(serialize_as='base64+meta')

        for dtype in (num.int, num.float):
            a = A(arr=num.zeros((3, 3), dtype=dtype))
            b = load_string(a.dump())
            assert a.arr.shape == b.arr.shape
            self.assertTrue(num.all(a.arr == b.arr))

        b_int = load_string


    def testPO(self):
        class SKU(StringPattern):
            pattern = '\\d{3}-[A-Z]{2}'

        class Comment(String):
            xmltagname = 'comment'

        class Quantity(Int):
            pass

        class USAddress(Object):
            country = String.T(
                default='US', optional=True, xmlstyle='attribute')
            name = String.T()
            street = String.T()
            city = String.T()
            state = String.T()
            zip = Float.T()

        class Item(Object):
            part_num = SKU.T(xmlstyle='attribute')
            product_name = String.T()
            quantity = Quantity.T()
            us_price = Float.T(xmltagname='USPrice')
            comment = Comment.T(optional=True)
            ship_date = DateTimestamp.T(optional=True)

        class Items(Object):
            item_list = List.T(Item.T())

        class PurchaseOrderType(Object):
            order_date = DateTimestamp.T(optional=True, xmlstyle='attribute')
            ship_to = USAddress.T()
            bill_to = USAddress.T()
            comment = Comment.T(optional=True)
            items = Items.T()

        class PurchaseOrder(PurchaseOrderType):
            xmltagname = 'purchaseOrder'

        xml = '''<?xml version="1.0"?>
<purchaseOrder orderDate="1999-10-20">
   <shipTo country="US">
      <name>Alice Smith</name>
      <street>123 Maple Street</street>
      <city>Mill Valley</city>
      <state>CA</state>
      <zip>90952</zip>
   </shipTo>
   <billTo country="US">
      <name>Robert Smith</name>
      <street>8 Oak Avenue</street>
      <city>Old Town</city>
      <state>PA</state>
      <zip>95819</zip>
   </billTo>
   <comment>Hurry, my lawn is going wild</comment>
   <items>
      <item partNum="872-AA">
         <productName>Lawnmower</productName>
         <quantity>1</quantity>
         <USPrice>148.95</USPrice>
         <comment>Confirm this is electric</comment>
      </item>
      <item partNum="926-AA">
         <productName>Baby Monitor</productName>
         <quantity>1</quantity>
         <USPrice>39.98</USPrice>
         <shipDate>1999-05-21</shipDate>
      </item>
   </items>
</purchaseOrder>
'''
        po1 = load_xml_string(xml)
        po2 = load_xml_string(po1.dump_xml())

        self.assertEqual(po1.dump(), po2.dump())

    def testDumpLoad(self):

        from tempfile import NamedTemporaryFile as NTF

        class A(Object):
            xmltagname = 'a'
            p = Int.T()

        a1 = A(p=33)
        an = [a1, a1, a1]

        def check1(a, b):
            self.assertEqual(a.p, b.p)

        def checkn(a, b):
            for ea, eb in zip(a, b):
                self.assertEqual(ea.p, eb.p)

        for (a, xdump, xload, check) in [
                (a1, dump, load, check1),
                (a1, dump_xml, load_xml, check1),
                (an, dump_all, load_all, checkn),
                (an, dump_all, iload_all, checkn),
                (an, dump_all_xml, load_all_xml, checkn),
                (an, dump_all_xml, iload_all_xml, checkn)]:

            for header in (False, True, 'custom header'):
                # via string
                s = xdump(a, header=header)
                b = xload(string=s)
                check(a, b)

                # via file
                f = NTF()
                xdump(a, filename=f.name, header=header)
                b = xload(filename=f.name)
                check(a, b)
                f.close()

                # via stream
                f = NTF()
                xdump(a, stream=f, header=header)
                f.seek(0)
                b = xload(stream=f)
                check(a, b)
                f.close()

        b1 = A.load(string=a1.dump())
        check1(a1, b1)

        f = NTF()
        a1.dump(filename=f.name)
        b1 = A.load(filename=f.name)
        check1(a1, b1)
        f.close()

        f = NTF()
        a1.dump(stream=f)
        f.seek(0)
        b1 = A.load(stream=f)
        check1(a1, b1)
        f.close()

        b1 = A.load_xml(string=a1.dump_xml())
        check1(a1, b1)

        f = NTF()
        a1.dump_xml(filename=f.name)
        b1 = A.load_xml(filename=f.name)
        check1(a1, b1)
        f.close()

        f = NTF()
        a1.dump_xml(stream=f)
        f.seek(0)
        b1 = A.load_xml(stream=f)
        check1(a1, b1)
        f.close()


def makeBasicTypeTest(Type, sample, sample_in=None, xml=False):

    if sample_in is None:
        sample_in = sample

    def basicTypeTest(self):

        class X(Object):
            a = Type.T()
            b = Type.T(optional=True)
            c = Type.T(default=sample)
            d = List.T(Type.T())
            e = Tuple.T(1, Type.T())
            xmltagname = 'x'

        x = X(a=sample_in, e=(sample_in,))
        x.d.append(sample_in)
        if sample_in is not sample:
            with self.assertRaises(ValidationError):
                x.validate()

        x.validate(regularize=sample_in is not sample)
        self.assertEqualNanAware(sample, x.a)

        if not xml:
            x2 = load_string(x.dump())

        else:
            x2 = load_xml_string(x.dump_xml())

        self.assertEqualNanAware(x.a, x2.a)
        self.assertIsNone(x.b)
        self.assertIsNone(x2.b)
        self.assertEqualNanAware(sample, x.c)
        self.assertEqualNanAware(sample, x2.c)
        self.assertTrue(isinstance(x.d, list))
        self.assertTrue(isinstance(x2.d, list))
        self.assertEqualNanAware(x.d[0], sample)
        self.assertEqualNanAware(x2.d[0], sample)
        self.assertEqual(len(x.d), 1)
        self.assertEqual(len(x2.d), 1)
        self.assertTrue(isinstance(x.e, tuple))
        self.assertTrue(isinstance(x2.e, tuple))
        self.assertEqualNanAware(x.e[0], sample)
        self.assertEqualNanAware(x2.e[0], sample)
        self.assertEqual(len(x.e), 1)
        self.assertEqual(len(x2.e), 1)

    return basicTypeTest


for Type in samples:
    for isample, sample in enumerate(samples[Type]):
        for xml in (False, True):
            setattr(GutsTestCase, 'testBasicType' + Type.__name__ +
                    str(isample) + ['', 'XML'][xml],
                    makeBasicTypeTest(Type, sample, xml=xml))

for Type in regularize:
    for isample, (sample_in, sample) in enumerate(regularize[Type]):
        for xml in (False, True):
            setattr(GutsTestCase, 'testBasicTypeRegularize' + Type.__name__ +
                    str(isample) + ['', 'XML'][xml],
                    makeBasicTypeTest(
                        Type, sample, sample_in=sample_in, xml=xml))


if __name__ == '__main__':
    unittest.main()

from __future__ import division, print_function, absolute_import
import unittest

from .. import common
import numpy as num

from pyrocko import util
from pyrocko.io import ims


def filled_int32(n, v):
    values = num.zeros(n, dtype=num.int32)
    values.fill(v)
    return values


class IMSTestCase(unittest.TestCase):

    def test_cm6(self):
        from pyrocko import ims_ext
        a1 = num.random.randint(-2**31, 2**31-1, 10000).astype(num.int32)
        a2 = num.random.randint(-2**31, 2**31-1, 10000).astype(num.int32)[::2]

        for values in [
                a1,
                a2,
                filled_int32(10, 0),
                filled_int32(10, 2**31),
                filled_int32(10, 2**31-1),
                num.zeros(0, dtype=num.int32)]:

            s = ims_ext.encode_cm6(values)
            values2 = ims_ext.decode_cm6(s, 0)
            assert values.size == values2.size
            assert num.all(values == values2)

    def test_checksum(self):
        from pyrocko import ims_ext
        # a = num.random.randint(-2**31, 2**31-1, 10000).astype(num.int32)
        m = 100000000
        vs = [m, m-1, m+1, 0, -1, 1, -m, -m-1, -m+1, 2**31-1]
        for v1 in vs:
            for v2 in vs:
                a = num.array([v1, v2], dtype=num.int32)
                assert ims_ext.checksum(a) == ims_ext.checksum_ref(a)

    @unittest.skip('known problem with checksum == -2^31')
    def test_checksum_nasty(self):
        vs = [-2**31]
        for v in vs:
            a = num.array([v], dtype=num.int32)
            assert ims_ext.checksum(a) == ims_ext.checksum_ref(a)

    def test_read_write(self):
        fns = [
            'test.iris.channel.ims',
            'test.iris.response.ims',
            'test.iris.station.ims',
            'test.iris.waveform-nodata.ims',
            'test.norsar.gse2',
            'test.cndc.waveform.gse2',
            'test.isc.event.with_headers.ims']

        fpaths = []
        for fn in fns:
            fpath = common.test_data_file(fn)
            fpaths.append(fpath)

        for sec in ims.iload(fpaths):
            if isinstance(sec, ims.WID2Section):
                sec.pyrocko_trace()

        s = ims.dump_string(ims.iload(fpaths))
        for sec in ims.iload_string(s):
            if isinstance(sec, ims.WID2Section):
                sec.pyrocko_trace()

    def test_ref_example1(self):
        s = b'''DATA_TYPE WAVEFORM GSE2.1:CM6'''
        version_dialect = ['GSE2.1', None]
        d = ims.DataType.deserialize(s, version_dialect)
        s2 = d.serialize(version_dialect)
        assert s == s2

    def test_ref_example2(self):
        s = b'''
BEGIN GSE2.1
MSG_TYPE DATA
MSG_ID 1040 GSE_IDC
REF_ID 9733 ANY_NDC
DATA_TYPE LOG GSE2.1
Command waveform processed.
STOP
'''
        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip()

    def test_ref_example3(self):
        s = b'''
BEGIN GSE2.1
MSG_TYPE DATA
MSG_ID 1243 GSE_IDC
REF_ID 1040 ANY_NDC
DATA_TYPE ERROR_LOG GSE2.1
An error was detected in the following request message:
   BEGIN GSE2.1
   MSG_TYPE request
   MSG_ID 1040 ANY_NDC
   TIME 94/03/01 TO 94/03/02
   *** Unrecognized time format ***
   STA_LIST ARA0
   WAVEFORM
   STOP
STOP
'''
        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip(), (s, s2)

    def test_ref_example4(self):
        s = b'''
BEGIN GSE2.1
MSG_TYPE DATA
MSG_ID 1040 GSE_IDC
REF_ID 5493 ANY_NDC
DATA_TYPE FTP_LOG GSE2.1
FTP_FILE pidc.org USER /pub/data/ANY_NDC 1994125001.msg.gz
STOP
'''
        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip()

    def test_ref_example5(self):
        s = b'''
DATA_TYPE WAVEFORM GSE2.1:CM6
OUT2 1996/10/15 09:56:00.000 KAF   shz           60.000
STA2 IDC_SEIS   62.11270   26.30621 WGS-84       0.195 0.014
OUT2 1996/10/15 09:56:00.000 KAF   shz           60.000
STA2 IDC_SEIS   62.11270   26.30621 WGS-84       0.195 0.014
'''
        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip()

    def test_ref_example6(self):
        s = b'''
DATA_TYPE CHANNEL GSE2.1
Net       Sta  Chan Aux   Latitude Longitude  Coord Sys       Elev   Depth   Hang  Vang Sample Rate Inst      On Date    Off Date
IDC_SEIS  ARA0  she       69.53490   25.50580 WGS-84       0.403 0.010   90.0  90.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARA0  shn       69.53490   25.50580 WGS-84       0.403 0.011    0.0  90.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARA0  shz       69.53490   25.50580 WGS-84       0.403 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARA1  shz       69.53630   25.50710 WGS-84       0.411 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARA2  shz       69.53380   25.50780 WGS-84       0.392 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARA3  shz       69.53460   25.50190 WGS-84       0.402 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARB1  shz       69.53790   25.50790 WGS-84       0.414 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARB2  shz       69.53570   25.51340 WGS-84       0.397 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARB3  shz       69.53240   25.51060 WGS-84       0.376 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARB4  shz       69.53280   25.49980 WGS-84       0.378 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARB5  shz       69.53630   25.49850 WGS-84       0.405 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARC1  shz       69.54110   25.50790 WGS-84       0.381 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARC2  she       69.53830   25.52290 WGS-84       0.395 0.010   90.0  90.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARC2  shn       69.53830   25.52290 WGS-84       0.395 0.010    0.0  90.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARC2  shz       69.53830   25.52290 WGS-84       0.395 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARC3  shz       69.53290   25.52310 WGS-84       0.376 0.010   -1.0   0.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARC4  she       69.52930   25.51170 WGS-84       0.377 0.010   90.0  90.0   40.000000 GS-13   1987/09/30
IDC_SEIS  ARC4  shn       69.52930   25.51170 WGS-84       0.377 0.010    0.0  90.0   40.000000 GS-13   1987/09/30
'''  # noqa
        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip()

    def test_ref_example7(self):
        s = b'''
DATA_TYPE NETWORK GSE2.1
Net       Description
IDC_SEIS  International Data Centre Seismic Network
IDC_HYDR  International Data Centre Hydroacoustic Network
'''
        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip()

    def test_ref_example8(self):
        s = b'''
DATA_TYPE STATION GSE2.1
Net       Sta   Type  Latitude  Longitude Coord Sys     Elev   On Date   Off Date
IDC_SEIS  ARCES hfa   69.53490   25.50580 WGS-84       0.403 1987/09/30
IDC_SEIS  ARA0  3C    69.53490   25.50580 WGS-84       0.403 1987/09/30
IDC_SEIS  ARA1  1C    69.53630   25.50710 WGS-84       0.411 1987/09/30
IDC_SEIS  ARA2  1C    69.53380   25.50780 WGS-84       0.392 1987/09/30
IDC_SEIS  ARA3  1C    69.53460   25.50190 WGS-84       0.402 1987/09/30
IDC_SEIS  ARB1  1C    69.53790   25.50790 WGS-84       0.414 1987/09/30
IDC_SEIS  ARB2  1C    69.53570   25.51340 WGS-84       0.397 1987/09/30
IDC_SEIS  ARB3  1C    69.53240   25.51060 WGS-84       0.376 1987/09/30
IDC_SEIS  ARB4  1C    69.53280   25.49980 WGS-84       0.378 1987/09/30
IDC_SEIS  ARB5  1C    69.53630   25.49850 WGS-84       0.400 1987/09/30
IDC_SEIS  ARC1  1C    69.54110   25.50790 WGS-84       0.381 1987/09/30
IDC_SEIS  ARC2  3C    69.53830   25.52290 WGS-84       0.395 1987/09/30
IDC_SEIS  ARC3  1C    69.53290   25.52310 WGS-84       0.376 1987/09/30
IDC_SEIS  ARC4  3C    69.52930   25.51170 WGS-84       0.377 1987/09/30
IDC_SEIS  ARC5  1C    69.53000   25.49820 WGS-84       0.374 1987/09/30
IDC_SEIS  ARC6  1C    69.53410   25.48820 WGS-84       0.395 1987/09/30
IDC_SEIS  ARC7  3C    69.53960   25.49360 WGS-84       0.362 1987/09/30
IDC_SEIS  ARD1  1C    69.54830   25.50930 WGS-84       0.395 1987/09/30
IDC_SEIS  ARD2  1C    69.54520   25.53080 WGS-84       0.366 1987/09/30
'''  # noqa
        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip()

    def test_ref_example9(self):
        s = b'''
DATA_TYPE BEAM GSE2.1
Bgroup   Sta  Chan Aux  Wgt     Delay
FIG1     FIA0  shz        1
FIG1     FIB1  shz        1
FIG1     FIB2  shz        0
FIG1     FIB3  shz        0
FIG1     FIB4  shz        1
FIG1     FIB5  shz        1
FIG1     FIC1  shz        1
FIG1     FIC2  shz        0
FIG1     FIC3  shz        0
FIG1     FIC4  shz        0
FIG1     FIC5  shz        1
FIG1     FIC6  shz        0
FIG2     FIA0  shn        1
FIG2     FIA0  she        1
FIG2     FIC7  shn        1
FIG2     FIC7  she        1

BeamID       Bgroup Btype R  Azim  Slow Phase       Flo    Fhi  O Z F    On Date    Off Date
FICB.01      FIG1     coh n  30.0 0.090 -          3.50   5.50  3 y BP 1997/01/01
FICB.02      FIG1     coh n  90.0 0.090 -          3.50   5.50  3 y BP 1997/01/01
FIIB.01      FIG2     inc n   0.0 0.000 -          8.00  16.00  3 y BP 1997/01/01
FICB.Pa      FIG1     coh n  -1.0 0.125 P          0.50  12.00  3 y BP 1997/01/01
FIIB.Sa      FIG1     inc n  -1.0 0.222 S          2.00   4.00  3 y BP 1997/01/01
FIIB.Lga     FIG2     inc n  -1.0 0.250 Lg         2.00   4.00  3 y BP 1997/01/01
FICB.Pb      FIG1     coh n  -1.0 0.125 P          0.50  12.00  3 y BP 1997/01/01
FIIB.Sb      FIG1     inc n  -1.0 0.222 S          2.00   4.00  3 y BP 1997/01/01
FIIB.Lgb     FIG2     inc n  -1.0 0.250 Lg         2.00   4.00  3 y BP 1997/01/01
'''  # noqa
        s2 = ims.write_string(ims.iload_string(s))
        for a, b in zip(s.strip().splitlines(), s2.strip().splitlines()):
            if a != b:
                print(a)
                print(b)

        assert s.strip() == s2.strip()

    def test_ref_example10(self):
        s = b'''
DATA_TYPE RESPONSE GSE2.1
CAL2 MIAR  BHZ      CMG-3N  4.11000000e+00  16.000    40.00000 1992/09/23 20:00
 (USNSN station at Mount Ida, Arkansas, USA)
PAZ2  1 V  7.29000000e+04         1.000   6   3 CMG-3 (NSN) Acc-Vel (Std)
 -3.14000000e-02  3.14000000e-04
 -1.97000000e-01  1.97000000e-03
 -2.01000000e+02  2.01000000e+00
 -6.97000000e+02  6.97000000e+00
 -7.54000000e+02  7.54000000e+00
 -1.05000000e+03  1.05000000e+01
  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00
 (Theoretical response provided by Guralp Systems, Ltd.)
DIG2  2  4.18000000e+05  5120.00000                   Quanterra QX80
FIR2  3   1.00e+00   16    0.006 C   32          QDP380/900616 stage 1
 -1.11328112e-03 -1.00800209e-03 -1.35286082e-03 -1.73045369e-03 -2.08418001e-03
 -2.38537718e-03 -2.60955630e-03 -2.73352256e-03 -2.73316190e-03 -2.58472445e-03
 -2.26411712e-03 -1.74846814e-03 -1.01403310e-03 -3.51681737e-05  1.23782025e-03
  3.15983174e-03  6.99944980e-03  9.09959897e-03  1.25423642e-02  1.63123012e-02
  2.02632397e-02  2.43172608e-02  2.84051094e-02  3.24604138e-02  3.64142842e-02
  4.01987396e-02  4.37450483e-02  4.69873249e-02  4.98572923e-02  5.22795729e-02
  5.41139580e-02  5.43902851e-02
FIR2  4   1.00e+00    4    0.077 C   36          QDP380/900616 stage 2
  1.50487336e-04  3.05924157e-04  4.42948687e-04  3.87117383e-04 -4.73786931e-05
 -9.70771827e-04 -2.30317097e-03 -3.70637676e-03 -4.62504662e-03 -4.46480140e-03
 -2.86984467e-03  7.00860891e-06  3.38519946e-03  6.00352836e-03  6.55093602e-03
  4.25995188e-03 -5.76023943e-04 -6.43416447e-03 -1.09213749e-02 -1.16364118e-02
 -7.26515194e-03  1.53727445e-03  1.19331051e-02  1.96156967e-02  2.03516278e-02
  1.18680289e-02 -4.64369030e-03 -2.41125356e-02 -3.86382937e-02 -3.98499220e-02
 -2.18683947e-02  1.61612257e-02  6.89623653e-02  1.26003325e-01  1.74229354e-01
  2.01834172e-01
FIR2  5   1.00e+00    2    0.379 C   32          QDP380/900616 stage 3,4,5
  2.88049545e-04  1.55313976e-03  2.98230513e-03  2.51714466e-03 -5.02926821e-04
 -2.81205843e-03 -8.08708369e-04  3.21542984e-03  2.71266000e-03 -2.91550322e-03
 -5.09429071e-03  1.33933034e-03  7.40034366e-03  1.82796526e-03 -8.81958286e-03
 -6.56719319e-03  8.38608573e-03  1.24268681e-02 -5.12978853e-03 -1.84868593e-02
 -1.79236766e-03  2.33604181e-02  1.30477296e-02 -2.51709446e-02 -2.93134767e-02
  2.12669298e-02  5.21898977e-02 -6.61517353e-03 -8.83535221e-02 -3.66062373e-02
  1.86273292e-01  4.03764486e-01
'''  # noqa

        s2 = ims.write_string(ims.iload_string(s))
        for a, b in zip(s.strip().splitlines(), s2.strip().splitlines()):
            if a != b:
                print(a)
                print(b)
                print()

        assert s.strip() == s2.strip()

    def test_ref_example11(self):

        s = b'''
DATA_TYPE OUTAGE GSE2.1
Report period from 1994/12/24 00:00:00.000 to 1994/12/25 12:00:00.000
NET       Sta  Chan Aux      Start Date Time          End Date Time        Duration Comment
IDC_SEIS  APL   shz      1994/12/24 08:13:05.000 1994/12/24 08:14:10.000     65.000
IDC_SEIS  APL   shn      1994/12/25 10:00:00.000 1994/12/25 10:00:00.030      0.030
'''  # noqa

        s2 = ims.write_string(ims.iload_string(s))
        assert s.strip() == s2.strip()


if __name__ == "__main__":
    util.setup_logging('test_ims', 'warning')
    unittest.main()

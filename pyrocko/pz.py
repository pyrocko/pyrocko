import scipy.signal as sig
import numpy as num
import sys
import gmtpy

class SacPoleZeroError(Exception):
    pass

def read_sac_zpk(filename):
    '''Read SAC Pole-Zero file.
    
       Returns (zeros, poles, constant).
    '''
    
    f = open(filename, 'r')
    sects = ('ZEROS', 'POLES', 'CONSTANT')
    sectdata = {'ZEROS': [], 'POLES': []}
    npoles = 0
    nzeros = 0
    constant = 1.0
    atsect = None
    for iline, line in enumerate(f):
        toks = line.split()
        if len(toks) == 0: continue
        if toks[0][0] == '#': continue
        if len(toks) != 2:
            f.close()
            raise SacPoleZeroError('Expected 2 tokens in line %i of file %s' % (iline+1, filename))
        
        if toks[0].startswith('*'): continue
        lsect = toks[0].upper()
        if lsect in sects:
            atsect = lsect
            sectdata[atsect] = []
            if lsect.upper() == 'ZEROS':
                nzeros = int(toks[1])
            elif toks[0].upper() == 'POLES':
                npoles = int(toks[1])
            elif toks[0].upper() == 'CONSTANT':
                constant = float(toks[1])
        else:
            if atsect:
                sectdata[atsect].append(complex(float(toks[0]), float(toks[1])))
    f.close()
    
    poles = sectdata['POLES']
    zeros = sectdata['ZEROS']
    npoles_ = len(poles)
    nzeros_ = len(zeros)
    if npoles_ > npoles:
        raise SacPoleZeroError('Expected %i poles but found %i in pole-zero file "%s"' % (npoles, npoles_, filename))
    if nzeros_ > nzeros:
        raise SacPoleZeroError('Expected %i zeros but found %i in pole-zero file "%s"' % (nzeros, nzeros_, filename))
    
    if npoles_ < npoles: poles.extend([complex(0.)]*(npoles-npoles_))
    if nzeros_ < npoles: zeros.extend([complex(0.)]*(nzeros-nzeros_))
    
    if len(poles) == 0 and len(zeros) == 0:
        raise SacPoleZeroError('No poles and zeros found in file "%s"' % (filename))
    
    if not num.all(num.isfinite(poles)):
        raise SacPoleZeroError('Not finite pole(s) found in pole-zero file "%s"' % (constant, filename))
    if not num.all(num.isfinite(zeros)):
        raise SacPoleZeroError('Not finite zero(s) found in pole-zero file "%s"' % (constant, filename))
    if not num.isfinite(constant):
        raise SacPoleZeroError('Ivalid constant (%g) found in pole-zero file "%s"' % (constant, filename))
    
    return zeros, poles, constant


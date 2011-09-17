import scipy
from scipy import signal
import time

import numpy

# Author: Nicolas Pinto
# Date: 2009/02

def fastncc(arr, filt):

    # -- integral images
    lisum = arr.cumsum(1).cumsum(0)
    lissq = (arr ** 2.).cumsum(1).cumsum(0)
    lh,lw = arr.shape
    fh,fw = filt.shape

    # -- denominator
    #s = time.time()
    rh,rw = lh-fh+1, lw-fw+1
    #cdef numpy.ndarray[double, ndim=2] div = numpy.zeros((rh,rw), dtype='float64')
    div = numpy.zeros((rh,rw), dtype='float64')
    #cdef double vsum, vssq
    #print lisum.shape
    #print rh, rw, lh, lw
    for j in xrange(rh-1):
        for i in xrange(rw-1):
            #print j, i, fh, fw
            #print j+fh, i+fw
            vsum = lisum[j+fh,i+fw] - lisum[j+fh,i] - lisum[j,i+fw] + lisum[j,i]
            vssq = lissq[j+fh,i+fw] - lissq[j+fh,i] - lissq[j,i+fw] + lissq[j,i]
            div[j,i] = (vssq - (vsum**2.))#/(fh*fw))

    div /= (fh*fw)

    div = div.clip(0, numpy.inf)**(1./2)
    div[div==0] = 1
    #div = sqrt(div)
    #e = time.time()
    #print "denom", e-s

    # -- numerator
    #s = time.time()
    filt -= filt.mean()
    filt /= scipy.linalg.norm(filt)
    #cdef numpy.ndarray[double, ndim=2] num = signal.fftconvolve(l,f[::-1,::-1], 'valid')
    num = signal.fftconvolve(arr, filt[::-1,::-1], 'valid')
    #e = time.time()
    #print "num", e-s
#
    # -- result
    #s = time.time()
    res = num/div
    #e = time.time()
    #print "res", e-s
    return res



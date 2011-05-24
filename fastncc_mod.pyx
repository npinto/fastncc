#from pylab import *
import scipy
from scipy import signal
import time

import numpy
cimport numpy
cimport cython

# Author: Nicolas Pinto
# Date: 2009/02

@cython.boundscheck(False)
def fastfncc(numpy.ndarray[double, ndim=2] l, numpy.ndarray[double, ndim=3] fb, Ffb=None):
    
    cdef double s, e    
    cdef int lh, lw, nf, fh, fw
    lh,lw = l.shape[0], l.shape[1]
    nf,fh,fw = fb.shape[0], fb.shape[1], fb.shape[2]
    cdef int n, fullh, fullw 
    fullh = l.shape[0]+fb[0].shape[0]-1
    fullw = l.shape[1]+fb[0].shape[1]-1

    # -- precompute filterbank fft's if needed
    if Ffb is None:
        Ffb = numpy.ndarray((nf,fullh,fullw), dtype='complex128')
        for n in xrange(nf):
            filt = fb[n]
            Ffb[n] = signal.fftn(filt[::-1,::-1], (fullh,fullw))

    # -- integral images
    cdef numpy.ndarray[double, ndim=2] lisum = numpy.zeros((lh+1,lw+1), 'float64')
    cdef numpy.ndarray[double, ndim=2] lissq = numpy.zeros((lh+1,lw+1), 'float64')
    cdef int j, i, lih, liw
    lih, liw = lisum.shape[0], lisum.shape[1]
    for j in xrange(1,lih):    
        for i in xrange(1,liw):
            lisum[j,i] = l[j-1,i-1] + lisum[j-1,i] + lisum[j,i-1] - lisum[j-1,i-1]
            lissq[j,i] = l[j-1,i-1]**2. + lissq[j-1,i] + lissq[j,i-1] - lissq[j-1,i-1]
    
    # -- denominator       
    cdef rh, rw
    rh,rw = lh-fh+1, lw-fw+1
    cdef numpy.ndarray[double, ndim=2] div = numpy.zeros((rh,rw), dtype='float64')
    cdef double vsum, vssq
    for j in xrange(rh):
        for i in xrange(rw):            
            vsum = lisum[j+fh,i+fw] - lisum[j+fh,i] - lisum[j,i+fw] + lisum[j,i]
            vssq = lissq[j+fh,i+fw] - lissq[j+fh,i] - lissq[j,i+fw] + lissq[j,i]
            div[j,i] = (vssq - (vsum**2.)/(fh*fw))

    div = div.clip(0,numpy.inf)**(1./2)

    # -- numerator and result
    cdef numpy.ndarray[double, ndim=3] res = numpy.empty((rh,rw,nf), dtype='float64')
    cdef numpy.ndarray[double, ndim=2] num
    Fimg = signal.fftn(l, (fullh,fullw))
    for n in xrange(nf):
        Ffilt = Ffb[n]
        num = numpy.real(signal.ifftn(Fimg*Ffilt))[fh-1:fh-1+rh,fw-1:fw-1+rw]
        res[:,:,n] = num/div

    # -- result
    return res


@cython.boundscheck(False)
def fastncc(numpy.ndarray[double, ndim=2] l, numpy.ndarray[double, ndim=2] f):

    cdef double s, e

    cdef int lh, lw, fh, fw
    lh,lw = l.shape[0], l.shape[1]
    fh,fw = f.shape[0], f.shape[1]
    
    # -- integral images
    s = time.time()
    cdef numpy.ndarray[double, ndim=2] lisum = numpy.zeros((lh+1,lw+1), 'float64')
    cdef numpy.ndarray[double, ndim=2] lissq = numpy.zeros((lh+1,lw+1), 'float64')
    cdef int j, i, lih, liw
    lih, liw = lisum.shape[0], lisum.shape[1]
    for j in range(1,lih):
        for i in range(1,liw):
            lisum[j,i] = l[j-1,i-1] + lisum[j-1,i] + lisum[j,i-1] - lisum[j-1,i-1]
            lissq[j,i] = l[j-1,i-1]**2. + lissq[j-1,i] + lissq[j,i-1] - lissq[j-1,i-1]
#             lisum[j,i] = ( l[<unsigned int>j-1,<unsigned int>i-1] 
#                            + lisum[<unsigned int>j-1,<unsigned int>i] 
#                            + lisum[<unsigned int>j,<unsigned int>i-1] 
#                            - lisum[<unsigned int>j-1,<unsigned int>i-1] )
#             lissq[j,i] = ( l[<unsigned int>j-1,<unsigned int>i-1]**2. 
#                            + lissq[<unsigned int>j-1,<unsigned int>i] 
#                            + lissq[<unsigned int>j,<unsigned int>i-1] 
#                            - lissq[<unsigned int>j-1,<unsigned int>i-1] )
    
    # -- denominator       
    cdef rh, rw
    rh,rw = lh-fh+1, lw-fw+1
    cdef numpy.ndarray[double, ndim=2] div = numpy.zeros((rh,rw), dtype='float64')
    cdef double vsum, vssq
    for j in range(rh):
        for i in range(rw):            
            vsum = lisum[j+fh,i+fw] - lisum[j+fh,i] - lisum[j,i+fw] + lisum[j,i]
            vssq = lissq[j+fh,i+fw] - lissq[j+fh,i] - lissq[j,i+fw] + lissq[j,i]
#             vsum = ( lisum[<unsigned int>j+fh,<unsigned int>i+fw] 
#                      - lisum[<unsigned int>j+fh,<unsigned int>i] 
#                      - lisum[<unsigned int>j,<unsigned int>i+fw] 
#                      + lisum[<unsigned int>j,<unsigned int>i] )
#             vssq = ( lissq[<unsigned int>j+fh,<unsigned int>i+fw] 
#                      - lissq[<unsigned int>j+fh,<unsigned int>i] 
#                      - lissq[<unsigned int>j,<unsigned int>i+fw] 
#                      + lissq[<unsigned int>j,<unsigned int>i] )
            div[j,i] = (vssq - (vsum**2.)/(fh*fw))

    div = div.clip(0,numpy.inf)**(1./2)
    #div = sqrt(div)
    e = time.time()
    print "denom", e-s

    # -- numerator
    s = time.time()
    f -= f.mean()
    f /= scipy.linalg.norm(f)
    cdef numpy.ndarray[double, ndim=2] num = signal.fftconvolve(l,f[::-1,::-1], 'valid')
    e = time.time()
    print "num", e-s

    # -- result
    s = time.time()
    res = num/div
    e = time.time()
    print "res", e-s
    return res



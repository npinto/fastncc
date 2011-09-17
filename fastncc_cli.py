from pylab import *
import time
import numpy

from naivencc_mod import *
#from fastncc_mod2 import *
from fastncc_mod import *

from scipy import *

print "go"
random.seed(1)
nf,fh,fw = 4,9,12
fb = randn(nf,fh,fw)
fb = fb.astype(double)
for n in xrange(nf):
    f = fb[n]
    f -= f.mean()
    f /= linalg.norm(f)
    fb[n] = f

l = lena()/255.
l = l[::8,::8]
#l = randn(480,620)
l = l.astype(double)

print 'naive'
s = time.time()
r1 = naivefncc(l,fb)
e = time.time()
tr1 = e-s
print tr1

print 'fast'
s = time.time()
r2 = fastfncc(l,fb)
e = time.time()
tr2 = e-s
print tr2

# - precomputed
print 'precomputed'
fullh = l.shape[0]+fb[0].shape[0]-1
fullw = l.shape[1]+fb[0].shape[1]-1
Ffb = numpy.ndarray((nf,fullh,fullw), dtype='complex128')
for n in xrange(nf):
    filt = fb[n]
    Ffb[n] = signal.fftn(filt[::-1,::-1], (fullh,fullw))
s = time.time()
r3 = fastfncc(l,fb,Ffb=Ffb)
e = time.time()
tr3 = e-s
print tr3

print tr1/tr2, linalg.norm(r1-r2)

print tr1/tr3, linalg.norm(r1-r3)

#matshow(r1)
#matshow(r2)

#show()


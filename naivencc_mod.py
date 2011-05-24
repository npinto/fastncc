from pylab import *
from scipy import *
import time

def naivencc(l, f):

    lh, lw = l.shape
    fh,fw = f.shape

    def ndot(patch,filt):
        patch = patch.ravel()
        patch -= patch.mean()
        patch /= linalg.norm(patch)
        #return linalg.norm(patch)
        return dot(patch,filt)
    
    rh, rw = lh-fh+1, lw-fw+1
    filt = f.ravel()
    filt -= filt.mean()
    filt /= linalg.norm(filt)
    ref = array([ndot(l[j:j+fh,i:i+fw], filt) 
                 for j in xrange(rh)
                 for i in xrange(rw)]).reshape((rh,rw))    
    return ref


def naivencc_denom(l, f):

    lh, lw = l.shape
    fh,fw = f.shape

    def ndot(patch,filt):
        patch = patch.ravel()
        patch -= patch.mean()
        return linalg.norm(patch)
    
    rh, rw = lh-fh+1, lw-fw+1
    filt = f.ravel()
    filt -= filt.mean()
    filt /= linalg.norm(filt)
    ref = array([ndot(l[j:j+fh,i:i+fw], filt) 
                 for j in xrange(rh)
                 for i in xrange(rw)]).reshape((rh,rw))    
    return ref


def naivefncc(l, fb):

    lh, lw = l.shape
    nf,fh,fw = fb.shape

    def ndot(patch,filt):
        patch = patch.ravel()
        patch -= patch.mean()
        patch /= linalg.norm(patch)
        #return linalg.norm(patch)
        return dot(patch,filt)
    
    rh, rw = lh-fh+1, lw-fw+1
    res = empty((rh,rw,nf))
    for n in xrange(nf):
        filt = fb[n].ravel()
        filt -= filt.mean()
        filt /= linalg.norm(filt)
        res[:,:,n] = array([ndot(l[j:j+fh,i:i+fw], filt) 
                            for j in xrange(rh)
                            for i in xrange(rw)]).reshape((rh,rw))    
        
    return res



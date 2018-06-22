import numpy as np
from scipy import signal

def make_data(kind, T, *args):
    def make_sine(T, freqs):
        if np.isscalar(freqs):
            return np.sin(freqs*np.arange(T)).reshape(T,1)
        
        y = np.zeros(T)
        for f in freqs:
            y += np.sin(f*np.arange(T))
        return y.reshape(T,1)

    def make_const(T, amp=1):
        return amp*np.ones((T,1))    
    
    def make_rect(T, f, duty=0.5, low=0, high=1):
        rect = signal.square(f*np.arange(T), duty)
        normalized = (rect+1)/2 * (high-low) + low
        return normalized.reshape(T,1)
        
    def make_deltas(T, f):
        rect = signal.square(f/2*np.arange(T))/2
        deltas = np.abs(np.diff(rect))
        deltas = np.insert(deltas, 0, 1)
        return deltas.reshape(T,1)

    switch = {'const'  : make_const,
              'sin'    : make_sine,
              'sine'   : make_sine,
              'rect'   : make_rect,
              'deltas' : make_deltas,
              }        
    return switch[kind.lower()](T, *args)


def combine_nets(nets, connType):
    pass


def augment_net(net, addN, connType):
    pass



def backprop_thru_time_loop(self,x,yTrue):
    """For validation. This is ~4x slower than using einsum"""
    T = len(x)
    y, h = self.feedforward(x)
    z, d, e = self.lagrange_mult(x, h, y, yTrue)  #TODO: abstract out the delta so I can use any loss function
    
    dSdWx = 0
    dSdWh = 0
    dSdWy = 0
    for t in range(T):
       dSdWh += np.outer( z[t] * self.fp( np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t]) ), h[t-1])
       dSdWx += np.outer( z[t] * self.fp( np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t]) ), x[t])
       dSdWy += np.outer( e[t], h[t])        
    dSdWx = -dSdWx/(T*self.tau)
    dSdWh = -dSdWh/(T*self.tau)
    dSdWy = -dSdWy/T 
    
    return dSdWx, dSdWh, dSdWy








   
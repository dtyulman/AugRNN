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












   
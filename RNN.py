"""Implemented based on notes from James Murray
See also: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import timeit 

class Loss:
    pass

class Nonlin:
    pass

def squared_error_loss(y, yTrue):
    T = len(y)
    return np.sum((y-yTrue)**2)/(2*T) #sums over T and Ny

np.seterr(over='raise')
def sech2(x):
    return (1./np.cosh(x))**2

class RNN(object):  
    def __init__(self, Nx, Nh, Ny, tau, f=np.tanh, fp=sech2):
        self.Nx, self.Nh, self.Ny = int(Nx), int(Nh), int(Ny)
        self.set_weights(*self.random_init_weights(Nx,Nh,Ny))
        self.tau = float(tau)
        self.f = f
        self.fp = fp
        self.h0 = np.zeros(self.Nh)
        #TODO: instead of Wx,Wh,Wy pass a single tuple (W_1, ..., W_R) to allow for arbitrary R-layer networks
        #TODO: abstract the cost function and the nonlin to separate classes

        
    def set_weights(self, Wx, Wh, Wy):
        if (self.Nh, self.Nx) != Wx.shape or (self.Nh, self.Nh) != Wh.shape or (self.Ny, self.Nh) != Wy.shape:
            raise ValueError('Dimensions of weight matrices must match number of neurons')       
        self.Wx, self.Wh, self.Wy = np.copy(Wx), np.copy(Wh), np.copy(Wy)

    
    def get_weights(self):
        return np.copy(self.Wx), np.copy(self.Wh), np.copy(self.Wy)    
          
    def get_size(self):
        return self.Nx, self.Nh, self.Ny    
    
    def random_init_weights(self, Nx, Nh, Ny):
        g = 1.5
        Wx = np.random.normal(0, np.sqrt(g/Nh), (Nh, Nx))
        Wh = np.random.normal(0, np.sqrt(g/Nh), (Nh, Nh))
        Wy = np.random.normal(0, np.sqrt(g/Nh), (Ny, Nh))
        
        return Wx,Wh,Wy
    
    
    def y_out(self, h):
        """Return y[t] given h[t]"""
        return np.dot(self.Wy, h)
    
    
    def feedforward(self, x):
        """
        x is T-by-Nx ndarray [x(1), ..., x(T)] with x(t) either scalar or vector i.e. shape of x(t) is (Nx,)
        y is T-by-Ny
        h is T-by-N
        """
        
        T = len(x)
        h = np.empty((T+1, self.Nh)) #allocate memory
        y = np.empty((T, self.Ny))
        
        h[-1] = self.h0 #put initial value at end so x,y,h time-idxs line up
        for t in xrange(T):
            h[t] = h[t-1] + ( -h[t-1] + self.f(np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t])) )/self.tau
            y[t] = self.y_out(h[t])
        return y, h
          
    
    def lagrange_mult(self, x, h, y, yTrue):
        T = len(x)
        z = np.empty((T+1, self.Nh)) #allocate memory
        d = np.empty((T+1, self.Nh))
        e = yTrue-y
        
        z[T-1] = np.dot(self.Wy.T, e[T-1]) #initialize 
        d[T-1] = np.zeros(self.Nh) #undefined 
        for t in xrange(T-2, -2, -1):           
            d[t] = z[t+1] * self.fp( np.dot(self.Wh,h[t])+np.dot(self.Wx,x[t+1]) )            
            z[t] = (1-1/self.tau)*z[t+1] + ( np.dot(self.Wh.T,d[t]) + np.dot(self.Wy.T,e[t]) )/self.tau 
        return z, d, e
    
    
    def backprop_thru_time(self, x, yTrue):
        T = len(x)
        y, h = self.feedforward(x)
        z, d, e = self.lagrange_mult(x, h, y, yTrue)  #TODO: abstract out the delta so I can use any loss function

        dSdWy = -np.einsum('ti,tj',e,h[0:-1])/T #don't sum over initial cond (last element in h)               
        dSdWh = -np.einsum('ti,tj',d,h)/(T*self.tau)        
        d = np.concatenate((d[-1].reshape(1,self.Nh), d[0:T-1]))
        dSdWx = -np.einsum('ti,tj',d,x)/(T*self.tau)
            
        return dSdWx, dSdWh, dSdWy
    
    
    def backprop_thru_time_loop(self,x,yTrue):
        """For validation""" #TODO: check if this is actually slower than using einsum
        T = len(x)
        y, h = self.feedforward(x)
        z, d, e = self.lagrange_mult(x, h, y, yTrue)  #TODO: abstract out the delta so I can use any loss function
        
        dSdWx = 0
        dSdWh = 0
        dSdWy = 0
        for t in range(T):
           dSdWh += np.outer( z[t] * self.fp( np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t]) ), h[t-1])
           dSdWx += np.outer( z[t] * self.fp( np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t]) ), x[t])
           dSdWy += np.outer(e[t], h[t])        
        dSdWx = -dSdWx/(T*self.tau)
        dSdWh = -dSdWh/(T*self.tau)
        dSdWy = -dSdWy/T 
        
        return dSdWx, dSdWh, dSdWy

    
    def sgd_step(self, x, yTrain, eta):
        dSdWx, dSdWh, dSdWy = self.backprop_thru_time(x, yTrain)          
        self.Wx -= eta[0]*dSdWx
        self.Wh -= eta[1]*dSdWh
        self.Wy -= eta[2]*dSdWy
        
                
    def sgd(self, x, yTrain, eta=0.05, epochs=1000, monitorLoss=True, monitorW=False):
        startTime = timeit.default_timer()
        
        if np.isscalar(eta):
            eta = (eta,eta,eta)
        if len(eta) != 3:
            raise ValueError('Learning rate must be scalar or length-3')
                
        if monitorLoss:
            loss = np.zeros(epochs)
        if monitorW:
            W = np.tile([np.zeros((self.Nh,self.Nx)), np.zeros((self.Nh,self.Nh)), np.zeros((self.Ny,self.Nh))], (epochs,1))                 
        for i in xrange(epochs):
            self.sgd_step(x, yTrain, eta)          
            
            if monitorLoss:
                yOut = self.feedforward(x)[0]
                loss[i] = squared_error_loss(yOut, yTrain) 
            if monitorW:
                W[i] = [self.Wx, self.Wh, self.Wy]
                        
        print('SGD: elapsed time: {} sec'.format(timeit.default_timer()-startTime)) 
        return loss
    
    
    def save(self, filename):
        raise NotImplemented
    
    
    @classmethod
    def load(cls, filename):
        raise NotImplemented



class AugRNN(RNN):
    """Augmentable RNN. Features:
        - adding neurons
        - freezing weights during training
        - reading out from a subnetwork of the hidden units
    """
        
    def __init__(self, Nx, Nh, Ny, tau, f=np.tanh, fp=sech2):
        super(AugRNN, self).__init__(Nx, Nh, Ny, tau, f, fp)
        self.freeze_weights(None,None,None)
        self.addNx=0
        self.addNh=0
        self.addNy=0        
        
    @classmethod
    def augment(cls, rnn, addNx=0, addNh=0, addNy=0):
        augRnn = cls(rnn.Nx+addNx, rnn.Nh+addNh, rnn.Ny+addNy, rnn.tau, rnn.f, rnn.fp)
        augRnn.addNx, augRnn.addNh, augRnn.addNy = addNx, addNh, addNy
        
        baseWx, baseWh, baseWy = rnn.get_weights()
        augRnn.set_weights(*augRnn.random_init_weights(augRnn.Nx,augRnn.Nh,augRnn.Ny))
        augRnn.set_subnetwork_weights(baseWx, baseWh, baseWy)        
        return augRnn
        
        
    def set_subnetwork_weights(self, Wx, Wh, Wy):
        """Sets the weights for the top-left corners of the weight matrices
        """
        Nhx, Nx  = Wx.shape
        Nh1, Nh2 = Wh.shape
        Ny,  Nhy = Wy.shape
        self.Wx[0:Nhx, 0:Nx], self.Wh[0:Nh1, 0:Nh2], self.Wy[0:Ny, 0:Nhy] = Wx, Wh, Wy
    
    
    def freeze_weights(self, freezeWx, freezeWh, freezeWy):
        self.freezeWx = freezeWx
        self.freezeWh = freezeWh
        self.freezeWy = freezeWy
         
           
    def sgd_step(self, x, yTrain, eta):
        dSdWx, dSdWh, dSdWy = self.backprop_thru_time(x, yTrain)
       
        if self.freezeWx is not None:
            dSdWx[self.freezeWx] = 0
        if self.freezeWh is not None:
            dSdWh[self.freezeWh] = 0
        if self.freezeWy is not None:
            dSdWy[self.freezeWy] = 0
        
        self.Wx -= eta[0]*dSdWx
        self.Wh -= eta[1]*dSdWh
        self.Wy -= eta[2]*dSdWy
       
       
    def feedforward_subnetwork(self, x, hIdx):
        tmpWy = np.copy(self.Wy) #TODO: this is mostly a hack, fix it
        self.Wy = np.zeros(tmpWy.shape)
        self.Wy[:,hIdx] = tmpWy[:,hIdx]
        y,h = self.feedforward(x)
        self.Wy = tmpWy
        return y, h
    
    
    def feedforward_base(self, x):
        raise NotImplemented
        
        
    def feedforward_new(self, x):
        raise NotImplemented
    
    

class RNNplotter():
    def __init_(self, rnn):
        self.rnn = rnn
            
        
    @staticmethod
    def plot_hidden(h, hIdx=None, color=['b'], ax=None, title=''):
        _,Nh = h.shape
        if ax == None:
            fig, ax = plt.subplots()        
        if hIdx == None: #plot everything by default
            hIdx = range(Nh)
        while len(color) < len(hIdx):
            color.append(color[-1])

        offset = 0
        for i in range(len(hIdx)):
            ax.plot(h[ 0:-1, hIdx[i] ]+offset, color=color[i])
            if i<len(hIdx)-1:
                max2min = np.max(h[ 0:-1, hIdx[i] ])  -  np.min(h[ 0:-1, hIdx[i+1] ])   
                offset += max2min*1.2 
        ax.set_xlabel('Time')
        ax.set_ylabel('Hidden unit')
        ax.set_title(title)
        return ax
    
    
    @staticmethod
    def plot_weights(Wx, Wh, Wy, title=''):        
        fig, ax = plt.subplots(2, 2)
        
#        wmin = np.min([Wx,Wh,Wy])                
#        wmax = np.max([Wx,Wh,Wy])
        
        mat = ax[0,1].matshow(Wh)
        ax[0,1].set_title('{}\nHidden'.format(title))
        ax[0,1].axis('off')
        
        ax[0,0].matshow(Wx)
        ax[0,0].set_title('Input')
        ax[0,0].axis('off')
        
        ax[1,1].matshow(Wy)
        ax[1,1].set_title('Output')
        ax[1,1].axis('off')
        
        ax[1,0].axis('off')
        plt.colorbar(mat, ax=ax[0,1])#vmin=wmin, vmax=wmax)

        
        
    @staticmethod
    def plot_full_base_new_readout(x, Nh, addNh, augRnn, title=''):
        fig, ax = plt.subplots(3,1)

        yFull, hAll = augRnn.feedforward(x)
        ax[0].plot(yFull)
        ax[0].plot(x)
        ax[0].legend(['y', 'x'])
        ax[0].set_title('{}\nFull network, $N_h$={}'.format(title,Nh+addNh))
        
        hBaseIdx = range(Nh)
        yBase,_ = augRnn.feedforward_subnetwork(x, hBaseIdx)
        ax[1].plot(yBase)
        ax[1].plot(x)
        ax[1].set_title('Base subnetwork $N_h$={}'.format(Nh))
        
        hNewIdx = range(Nh, Nh+addNh)
        yNew,_ = augRnn.feedforward_subnetwork(x, hNewIdx)
        ax[2].plot(yNew)
        ax[2].plot(x)
        ax[2].set_title('New subnetwork $N_h$={}'.format(addNh))
        
        fig.tight_layout()
        
        return yFull, yBase, yNew, hAll
    
        
        
        
       
        
        
 















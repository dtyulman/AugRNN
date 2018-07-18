"""Implemented based on notes from James Murray
See also: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
"""
import numpy as np
np.seterr(over='raise')

import matplotlib.pyplot as plt
from custom_utils import Timer
import cPickle


class Loss: #TODO
    pass


class NonLin: #TODO
    def __init__(self, nonlin):
        pass    
    f = np.tanh
    def fp(self, x): #sech^2(x):
        return 1.-np.tanh(x)**2   


class L2(object):
    def __init__(self, lmbda):
        self.lmbda = lmbda
        
    def f(self, Wx, Wh, Wy):
        """Takes in all the weights from the RNN, returns a scalar penalty"""
        return 0.5*self.lmbda*np.sum([np.sum(Wx**2), np.sum(Wh**2), np.sum(Wy**2)])
    
    def fp(self, Wx, Wh, Wy):
        """Takes in all the weights from the RNN, returns the regularization term for each weight"""
        return self.lmbda*Wx, self.lmbda*Wh, self.lmbda*Wy
        
    
def weights_to_vector(Wx,Wh,Wy):
    W = np.concatenate((Wx.flatten(), Wh.flatten(), Wy.flatten()))
    return W


def vector_to_weights(W, Nx, Nh, Ny):
    l = 0 #pointer to leftmost entry of Wx
    r = Nx*Nh #rightmost
    Wx = W[l:r].reshape(Nh, Nx)
    
    l = r #leftmost of Wh
    r = r + Nh*Nh
    Wh = W[l:r].reshape(Nh, Nh)
    
    l = r #leftmost of Wy
    r = r + Ny*Nh
    Wy = W[l:r].reshape(Ny, Nh)
    
    return Wx, Wh, Wy


def numerical_gradient(f, x, eps=1e-6):   
    """f is f(x), x=[x1,..,xN] is a vector"""
    g = np.full((f(x).size, x.size), np.nan) #TODO: switch to np.empty after testing
    for i in xrange(x.size):
        eps_i = np.zeros(x.size)
        eps_i[i] = eps/2.           
        g[:,i] = (f(x+eps_i) - f(x-eps_i))/eps
    return g.squeeze()
    

class HessEWC(object):
    """
    Similar to Elastic Weight Consolidation (Kirkpatrick et al. 2017), 
    but uses Hessian insead of Fisher Info.
    """
    #TODO: this computes the entire Hessian directly, which uses a ton of memory. Use Hessian-free method.
    def __init__(self, lmbda, rnn, data, H=None):
        self.lmbda = lmbda
        self.rnn = rnn
        self.Wx_old, self.Wh_old, self.Wy_old = rnn.get_weights()
        self.W_old = weights_to_vector(self.Wx_old, self.Wh_old, self.Wy_old)
        if H is None:
            self.H = self.hessian(rnn, data.x, data.y)
        else:
            self.H = H #for debugging, pass in the hessian

            
    def vector_to_weights(self, W):
        return vector_to_weights(W, self.rnn.Nx, self.rnn.Nh, self.rnn.Ny)
            
            
    def hessian(self, rnn, x, yTrue):                      
        return self.hessian_outer_product_approx(rnn, x, yTrue)
    
    
    def hessian_numerical(self, rnn, x, yTrue):   
        def dSdW(W):
            Wx,Wh,Wy=self.vector_to_weights(W)
            perturbRnn = RNN(Wx,Wh,Wy, rnn.tau, rnn.f, rnn.fp)
            dSdWx, dSdWh, dSdWy = perturbRnn.backprop_thru_time(x,yTrue)
            return weights_to_vector(dSdWx, dSdWh, dSdWy)
        
        with Timer('Numerical Hessian'):
            H = numerical_gradient(dSdW, self.W_old)         
        return H
        
    
    def hessian_outer_product_approx(self, rnn, x, yTrue):
        y,_ = rnn.feedforward(x)
        T = len(y)
        e = y-yTrue
        
        n = rnn.Nh*rnn.Nx + rnn.Nh*rnn.Nh + rnn.Nh*rnn.Ny 
        H = np.zeros((n,n))
        dLdWx,dLdWh,dLdWy = rnn.backprop_thru_time_double_loop(x, yTrue)
        for t in xrange(T):
            dLdW = weights_to_vector(dLdWx[t], dLdWh[t], dLdWy[t])
            H += np.outer(dLdW, dLdW)/e[t]**2 #TODO: assumes e[t] is scalar        
        return H/T
    
    
    def hessian_diag_approx(self, rnn): 
        #TODO
        pass
    
    
    def f(self, Wx, Wh, Wy):
        diff = weights_to_vector(Wx,Wh,Wy)-self.W_old
        L = self.lmbda/2 * np.sum( np.dot(diff, np.dot(self.H, diff)) )
        return L
        
    def fp(self, Wx, Wh, Wy):    
        return self.vector_to_weights( self.lmbda*np.dot(self.H, weights_to_vector(Wx,Wh,Wy)-self.W_old) )
 
            
def sech2(x):
    return 1.-np.tanh(x)**2
  

class RNN(object):  
    def __init__(self, Nx, Nh, Ny, tau, f=np.tanh, fp=sech2, reg=None, name=''):
        if np.isscalar(Nx) and np.isscalar(Nh) and np.isscalar(Ny):
            self.Nx, self.Nh, self.Ny = int(Nx), int(Nh), int(Ny)
            self.set_weights(*self.random_init_weights(Nx,Nh,Ny))
        else:
            Wx, Wh, Wy = Nx, Nh, Ny
            Nhx, Nx = Wx.shape
            Nh, Nh1 = Wh.shape 
            Ny, Nhy = Wy.shape
            if (Nhx!=Nh) or (Nh!=Nh1) or (Nh1!=Nhy):
                raise ValueError('Dimensions of weight matrices not consistent') 
            self.Nx, self.Nh, self.Ny = Nx, Nh, Ny
            self.set_weights(Wx, Wh, Wy)
        self.tau  = float(tau)
        self.f    = f   #nonlinearity
        self.fp   = fp  #derivative of nonlinearity
        self.reg  = reg #regularizer
        self.h0   = np.zeros(self.Nh)
        self.name = name #for plotting, using str(), etc
        self.freeze_weights(None,None,None)
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
        g = 1.5 #TODO don't hardcode this
        Wx = np.random.normal(0, np.sqrt(g/Nh), (Nh, Nx))
        Wh = np.random.normal(0, np.sqrt(g/Nh), (Nh, Nh))
        Wy = np.random.normal(0, np.sqrt(g/Nh), (Ny, Nh))
        
        return Wx,Wh,Wy
           
    
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
            y[t] = np.dot(self.Wy, h[t])
        return y, h
          
    
    def lagrange_mult(self, x, h, y, yTrue):
        T = len(x)
        z = np.empty((T+1, self.Nh)) #allocate memory
        d = np.empty((T+1, self.Nh))
        e = yTrue-y
        
        z[T-1] = np.dot(self.Wy.T, e[T-1]) #initialize 
        d[T-1] = np.zeros(self.Nh) #undefined 
        for t in reversed(xrange(-1, T-1)):          
            d[t] = z[t+1] * self.fp( np.dot(self.Wh,h[t])+np.dot(self.Wx,x[t+1]) )            
            z[t] = (1-1/self.tau)*z[t+1] + ( np.dot(self.Wh.T,d[t]) )/self.tau + np.dot(self.Wy.T,e[t])  
        return z, d, e
    
    
    def backprop_thru_time(self, x, yTrue):
        return self.backprop_thru_time_lagrange(x, yTrue)
        
        
    def backprop_thru_time_lagrange(self, x, yTrue):
            T = len(x)
            y, h = self.feedforward(x)
            z, d, e = self.lagrange_mult(x, h, y, yTrue) #TODO: abstract out the delta so I can use any loss function
    
            dSdWy = -np.einsum('ti,tj',e,h[0:-1])/T #don't sum over initial cond (last element in h)               
            dSdWh = -np.einsum('ti,tj',d,h)/(T*self.tau)        
            d = np.concatenate((d[-1].reshape(1,self.Nh), d[0:T-1]))
            dSdWx = -np.einsum('ti,tj',d,x)/(T*self.tau)           
            return dSdWx, dSdWh, dSdWy
    
    
    def backprop_thru_time_lagrange_loop(self,x,yTrue):
        """For validation. This is ~4x slower than using einsum"""
        T = len(x)
        y, h = self.feedforward(x)
        z, d, e = self.lagrange_mult(x, h, y, yTrue)  #TODO: abstract out the delta so I can use any loss function
        
        dSdWx = np.zeros(self.Wx.shape)
        dSdWh = np.zeros(self.Wh.shape)
        dSdWy = np.zeros(self.Wy.shape)
        for t in range(T):
           dSdWx += np.outer( z[t] * self.fp( np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t]) ), x[t])
           dSdWh += np.outer( z[t] * self.fp( np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t]) ), h[t-1])
           dSdWy += np.outer( e[t], h[t])        
        dSdWx = -dSdWx/(T*self.tau)
        dSdWh = -dSdWh/(T*self.tau)
        dSdWy = -dSdWy/T     
        return dSdWx, dSdWh, dSdWy
       
        
    def backprop_thru_time_chain(self, x, yTrue):
        """Equivalent (with some finite-precusion error) to backprop_thru_time_lagrange but follows the 
        more common chain-rule equations (e.g. see Deep Learning by Goodfellow et al.), instead of the
        Lagrange multiplier formalism (as in LeCun 1988). Note this is slower because of the for-loop 
        instead of einsum"""
        T = len(x)
        y, h = self.feedforward(x)
        e = y-yTrue
        
        dLdWx = np.zeros(self.Wx.shape)
        dLdWh = np.zeros(self.Wh.shape)
        dLdWy = np.zeros(self.Wy.shape)
    
        d = np.zeros(h[0].shape) #delta, i.e. delta_i(t) = dL/dh_i(t)
        for t in reversed(xrange(T)): 
            dLdWy += np.outer( e[t], h[t] )

            d += np.dot(self.Wy.T, e[t]) #d[t] is now complete here (after precomp from prev iter, see below) 
            fp = self.fp( np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t]) ) / self.tau         
            d_fp = d * fp
            
            dLdWh += np.outer( d_fp, h[t-1] )            
            dLdWx += np.outer( d_fp, x[t] )
            
            dhdh = np.matmul(self.Wh.T, np.diag(fp)) + np.eye(self.Nh)*(1-1/self.tau) #dh_j(t+1)/dh_i(t)
            d = np.dot(dhdh, d) #precomp part of d[t-1] (i.e. for next iter) 
            #d = np.dot(self.Wh.T, d_fp) + d * (1-1/self.tau) #note this is equivalent cf. z[t] in lagrange_mult()                 
           
        dLdWy = dLdWy/T 
        dLdWx = dLdWx/T
        dLdWh = dLdWh/T
        
        return dLdWx, dLdWh, dLdWy

    
    def backprop_thru_time_double_loop(self, x, yTrue):
        """Slower O(n^2), but explicitly computes E(t) for every t along the way, cf. O(n) for the standard bptt
        Note that the return values dLdWx,dLdWh, dLdWy are length-T lists (unlike the other bptt methods). To get 
        the full error, must do e.g. np.sum(dLdWx)/T"""
        T = len(x)
        y, h = self.feedforward(x)
        e = y-yTrue
        
        dLdWx = []
        dLdWh = []
        dLdWy = []        
        for t in xrange(T): 
            dLdWy.append( np.outer(e[t],h[t]) )
            
            dLdWx.append( np.zeros(self.Wx.shape) )
            dLdWh.append( np.zeros(self.Wh.shape) )
            d = np.dot(self.Wy.T, e[t]) #delta, i.e. delta_i(t') = dL(t)/dh_i(t')
            for tp in reversed(xrange(t+1)):
                fp = self.fp( np.dot(self.Wh,h[tp-1])+np.dot(self.Wx,x[tp]) ) / self.tau         
                d_fp = d * fp
                
                dLdWh[t] += np.outer( d_fp, h[tp-1] )            
                dLdWx[t] += np.outer( d_fp, x[tp] )
                
                dhdh = np.matmul(self.Wh.T, np.diag(fp)) + np.eye(self.Nh)*(1-1/self.tau) #dh_j(t'+1)/dh_i(t')
                d = np.dot(dhdh, d) #comp d[t-1] for next iter 
                
        return dLdWx, dLdWh, dLdWy
    

    def loss(self, y, yTrue): 
        """Squared error loss"""
        T = len(y)
        L = np.sum((y-yTrue)**2)/(2*T) #sums over T and Ny
        if self.reg is not None:
            L = L + self.reg.f(self.Wx, self.Wh, self.Wy)
        return L


    def action(self, y, h, z, yTrue):
        """Loss plus Lagrange multiplier to ensure continuous dynamics, averaged over time"""
                
        
    def freeze_weights(self, freezeWx=None, freezeWh=None, freezeWy=None):
        self.freezeWx = freezeWx
        self.freezeWh = freezeWh
        self.freezeWy = freezeWy
    
    
    def sgd_step(self, x, yTrain, eta):
        dSdWx, dSdWh, dSdWy = self.backprop_thru_time(x, yTrain) 
                
        if self.reg is not None:
            dRx, dRh, dRy = self.reg.fp(self.Wx, self.Wh, self.Wy)
            dSdWx, dSdWh, dSdWy = dSdWx+dRx, dSdWh+dRh, dSdWy+dRy
            
        if self.freezeWx is not None:
            dSdWx[self.freezeWx] = 0
        if self.freezeWh is not None:
            dSdWh[self.freezeWh] = 0
        if self.freezeWy is not None:
            dSdWy[self.freezeWy] = 0
        
        self.Wx -= eta[0]*dSdWx
        self.Wh -= eta[1]*dSdWh
        self.Wy -= eta[2]*dSdWy
        
        return dSdWx, dSdWh, dSdWy
        
                
    def sgd(self, x, yTrain, eta=0.05, epochs=1000, 
            monitorLoss=True, monitorW=False, monitorY=False, monitorH=False):        
        if np.isscalar(eta):
            eta = (eta,eta,eta)
            #TODO: adaptive learning rate
        if len(eta) != 3:
            raise ValueError('Learning rate must be scalar or length-3')
                
        loss = []
        yHist = []
        hHist = []
        W = []
        with Timer('SGD'):
            for epoch in xrange(epochs):
                dSdWx, dSdWh, dSdWy = self.sgd_step(x, yTrain, eta)          
                if monitorLoss:
                    y,h = self.feedforward(x)  #TODO: don't recompute. These values already exist from the sgd_step                 
                    loss.append( self.loss(y, yTrain) )
                if monitorY: #TODO: this will break if monitorY = True but monitorLoss = False
                    yHist.append(y)
                if monitorH:
                    hHist.append(h)
                if monitorW:
                    W.append( self.get_weights() )   
                    
                if epoch%100 == 0:
                    print "Epoch:{}, Loss:{}, Norm:{}, Grad:{}".format(
                            epoch, 
                            self.loss(y, yTrain), 
                            np.linalg.norm(weights_to_vector(self.Wx,self.Wh,self.Wy)),
                            np.linalg.norm(weights_to_vector(dSdWx, dSdWh, dSdWy)))
                    
        return loss, W, yHist, hHist
    
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {'weights': self.get_weights(),
                'tau': self.tau,
                'nonlin': self.f,
#                'nonlin_derivative': self.fp, #TODO: figure out how to save this properly 
                'regularizer': self.reg,
                'name': self.name
                }
        with open(filename, 'w') as f:
            cPickle.dump(data, f)
    
    
    @classmethod
    def load(cls, filename):
        """Load a neural network from the file ``filename``.  Returns an
        instance of RNN.
        """
        with open(filename, 'r') as f:
            data = cPickle.load(f)
        
        Wx, Wh, Wy = data['weights']
        tau = data['tau']
        nonlin = data['nonlin']
        nonlin_derivative = sech2 #data['nonlin_derivative'] #TODO
        reg = data['regularizer']
        name = data['name']
        
        rnn = cls(Wx, Wh, Wy, tau, f=nonlin, fp=nonlin_derivative, reg=reg, name=name) 
        return rnn


class SubRNN(RNN):
    """Subdividable RNN. Supports control over subnetworks of the larger RNN"""
    
    def __init__(self, *args, **kwargs):
        super(SubRNN, self).__init__(*args, **kwargs)
        self.subnets = np.zeros(self.Nh)  
    
    
    def def_subnet(self, hIdx, subnet):
        self.subnets[hIdx] = subnet
        
    
    def get_subnet_weights(self, subnet):
        subWx = self.Wx[self.subnets==subnet, :]
        subWh = self.Wh[np.ix_(self.subnets==subnet, self.subnets==subnet)]
        subWy = self.Wy[:,self.subnets==subnet]
        return subWx, subWh, subWy
        
    
    def get_subnet_size(self, subnet=None):
        if subnet is None: #return all by default
            _,cts = np.unique(self.subnets, return_counts=True)
            return cts
        return np.sum(self.subnets==subnet)
        
    
    def set_subnet_weights(self, subWx=None, subWh=None, subWy=None, subnet=0):
        if subWx is not None:
            self.Wx[self.subnets==subnet, :] = subWx
        if subWh is not None:
            self.Wh[np.ix_(self.subnets==subnet, self.subnets==subnet)] = subWh
        if subWy is not None:
            self.Wy[:, self.subnets==subnet] = subWy
        
        return subWx, subWh, subWy
            

    def feedforward_subnet(self, x, subnet):
        T = len(x)
        h = np.empty((T+1, self.Nh)) #allocate memory
        y = np.empty((T, self.Ny))
        
        h[-1] = self.h0 #put initial value at end so x,y,h time-idxs line up
        for t in xrange(T):
            h[t] = h[t-1] + ( -h[t-1] + self.f(np.dot(self.Wh,h[t-1])+np.dot(self.Wx,x[t])) )/self.tau
            y[t] = np.dot(self.Wy[:,self.subnets==subnet], h[t,self.subnets==subnet]) #this line is different from RNN.feedforward()
        return y, h
    


class AugRNN(SubRNN):
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
        
          

class MplPlotter():
    """Plots RNN results using matplotlib (mpl)"""    
    def __init__(self, rnn):
        self.rnn = rnn

    
    @staticmethod
    def plot_data(x, y, ax=None, title=''):                
        if ax is None:
            fig, ax = plt.subplots() 
        
        ax.plot(y, label='Output y')
        if x is not None:
            ax.plot(x, label='Input x')
            ax.legend()
        
        ax.set_title(title)
        
        plt.pause(0.2)
        return ax          
    
    
    def plot_hidden(self, h, hIdx=None, ax=None, title=''):
        if ax is None:
            fig, ax = plt.subplots()        
        if hIdx is None: #plot everything by default
            hIdx = range(self.rnn.Nh)
        
        offset = 0
        color = 'blue'
        for i in range(len(hIdx)):            
            try:
                if self.rnn.subnets[i] != self.rnn.subnets[i-1]:
                    color = ax._get_lines.get_next_color()
            except AttributeError:
                pass
                              
            ax.plot(h[ 0:-1, hIdx[i] ]+offset, color=color)
            
            if i<len(hIdx)-1:
                max2min = np.max(h[ 0:-1, hIdx[i] ])  -  np.min(h[ 0:-1, hIdx[i+1] ])   
                offset += max2min*1.2 
                
        ax.set_xlabel('Time')
        ax.set_ylabel('Hidden unit')
        ax.set_title('{} {}'.format(self.rnn.name, title))
        
        plt.pause(0.2)
        return ax
    
    
    def plot_weights(self, ax=None, title=''):        
        if ax is None:
            fig, ax = plt.subplots(2, 2)        
#        wmin = np.min([Wx,Wh,Wy])                
#        wmax = np.max([Wx,Wh,Wy])        
        mat = ax[0,1].matshow(self.rnn.Wh)
        ax[0,1].set_title('{}\nHidden'.format(title))
        ax[0,1].axis('off')
        
        ax[0,0].matshow(self.rnn.Wx)
        ax[0,0].set_title('Input')
        ax[0,0].axis('off')
        
        ax[1,1].matshow(self.rnn.Wy)
        ax[1,1].set_title('Output')
        ax[1,1].axis('off')
        
        ax[1,0].axis('off')
        plt.colorbar(mat, ax=ax[0,1]) #vmin=wmin, vmax=wmax)

        plt.pause(0.2)
        return ax
        
            
    def plot_subnet_readout(self, yFull, ySubs, axs=None, title=''):
        Nsubs = len(ySubs)
        if axs is None:
            fig, axs = plt.subplots(Nsubs+1,1)
            
        self.plot_data(None, yFull, 
                       title='{} {}\nFull network , $N_h$={}'.format(self.rnn.name, title, self.rnn.Nh), ax=axs[0])        
        i=1
        for k,y in ySubs.iteritems():   
            self.plot_data(None, y, 
                           title='Subnet#{} , $N_h$={}'.format(k, self.rnn.get_subnet_size(subnet=k)), ax=axs[i]) 
            i+=1                              
        return axs
    
    
    def plot_loss(self, loss, ax=None, title='', **kwargs):
        if ax is None:
            fig, ax = plt.subplots()            
        ax.plot(loss, **kwargs)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title( '{} {}\n{}'.format(self.rnn.name, title, 'Squared error loss') ) 
        #TODO: update this to str(self.rnn.loss) when/if I abstract loss into a separate class               
        plt.pause(0.2)
        return ax
    
    
    def plot_history(self, history, epochs=[], maxPlots=100):
        if epochs == []:
            epochs = np.linspace(0, len(history), min(len(history), maxPlots), endpoint=False, dtype=int)
            
        t = len(epochs)
        r = int(np.floor(np.sqrt(t)))
        c = int(np.ceil (np.sqrt(t)))
        fig, axs = plt.subplots(r,c)
        
        for i in range(t):
            ax = axs.ravel()[i]
            self.plot_data(None, history[epochs[i]], ax=ax, title='Epoch {}'.format(epochs[i]))
            ax.axis('off')
        
        fig.tight_layout()
#        plt.pause(0.2)
        return axs
        
        
class Data(object):
     """Container for training/validation/test data"""
     #TODO: investigate whether using __slots__ will make this better https://stackoverflow.com/questions/472000/usage-of-slots
     def __init__(self, **kwargs):
         self.__dict__.update(kwargs)
                 

class RNN_Manager(object):
    def __init__(self, rnn, PlotterClass=MplPlotter):
        self.rnn = rnn
        self.Winit = rnn.get_weights()
        self.plt = PlotterClass(self.rnn)
        self.train = Data() 
        self.test = Data()
        
        
    def train_network(self, x, y, plotLoss=False, **kwargs):
        if x is not None and y is not None:          
            self.train.x = x
            self.train.y = y
        self.loss, self.wHist, self.yHist, self.hHist = self.rnn.sgd(self.train.x, self.train.y, **kwargs)       
        self.reset_test_outputs() #clear the test outputs since they are no longer valid      
        if plotLoss:
            self.plt.plot_loss(self.loss)
      
        
    def reset_test_outputs(self):
        """Run this whenever self.rnn is modified in any way e.g. by training or manually setting weights"""
        try:
            self.test = Data(x=self.test.x, name=self.test.name)
        except AttributeError:
            self.test = Data()

        
    def set_test_input(self, x, name=''):
        self.test = Data(x=x, name=name)
        
    
    def plot_training_data(self, **kwargs):
        self.plt.plot_data(self.train.x, 
                           self.train.y, 
                           title='{} Training Data {}'.format(self.rnn.name,
                                                              kwargs.pop('title','')), 
                           **kwargs)
        
        
    def plot_hidden(self, **kwargs):
        try:
            self.test.h
        except:
            self.test.y, self.test.h = self.rnn.feedforward(self.test.x)
        ax = self.plt.plot_hidden(self.test.h, **kwargs)       
        return ax
        
    
    def plot_feedforward(self, **kwargs):
        try:
            self.test.y
        except:
            self.test.y, self.test.h = self.rnn.feedforward(self.test.x)   
        
        ax = self.plt.plot_data(self.test.x, self.test.y, 
                                title='{} {}\n Test: {}'.format(self.rnn.name, 
                                                                kwargs.pop('title', ''),
                                                                self.test.name), 
                           **kwargs)
        return ax
         
         
    def plot_feedforward_subnet(self, subnets=None, **kwargs):   
        if subnets is None:
            subnets = np.unique(self.rnn.subnets)
            
        for s in subnets:
            try: self.test.ySubs[s] #check if the outputs we want already exist
            except: #if not, compute them
                try: self.test.ySubs
                except: self.test.ySubs = {}
                self.test.ySubs[s],_  = self.rnn.feedforward_subnet(self.test.x, subnet=s)
        try: self.test.y
        except: self.test.y, self.test.h = self.rnn.feedforward(self.test.x)   
            
        axs = self.plt.plot_subnet_readout(self.test.y, self.test.ySubs, **kwargs)
        return axs

   
 















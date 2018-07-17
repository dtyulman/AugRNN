import numpy as np
from RNN import RNN, Data, numerical_gradient, weights_to_vector, vector_to_weights
from script_helpers import make_data
from custom_utils import Timer


rnn = RNN(1,50,1,10)
T = 400
f = 0.05
trainData = Data(x = make_data('rect', T, f, 0.1),
                 y = make_data('sin', T, [f, 2*f, 3*f]))

def E(W):
     Wx,Wh,Wy=vector_to_weights(W, 1,50,1)
     perturbRnn = RNN(Wx,Wh,Wy, rnn.tau)
     return perturbRnn.loss( perturbRnn.feedforward(trainData.x)[0], trainData.y )
 
def Et(W):
     Wx,Wh,Wy=vector_to_weights(W, 1,50,1)
     perturbRnn = RNN(Wx,Wh,Wy, rnn.tau)
     return perturbRnn.loss( perturbRnn.feedforward(trainData.x)[0], trainData.y )

 
with Timer('bptt'):
    dx, dh, dy = rnn.backprop_thru_time_chain(trainData.x, trainData.y)

with Timer('bptt double loop'):
    dx2_t, dh2_t, dy2_t = rnn.backprop_thru_time_double_loop(trainData.x, trainData.y)
    dx2 = sum(dx2_t)/T
    dh2 = sum(dh2_t)/T
    dy2 = sum(dy2_t)/T
    
with Timer('lagrange'):
    dxl, dhl, dyl = rnn.backprop_thru_time_lagrange(trainData.x, trainData.y)

with Timer ('lagrange loop'):
    dxll, dhll, dyll = rnn.backprop_thru_time_lagrange_loop(trainData.x, trainData.y)

#with Timer('numerical'):
#    dn = numerical_gradient(E, weights_to_vector(*rnn.get_weights()), 1e-6)
#    dxn, dhn, dyn = vector_to_weights( dn, rnn.Nx, rnn.Nh, rnn.Ny )

#with Timer('double loop numerical'):
#    dn = numerical_gradient(E, weights_to_vector(*rnn.get_weights()), 1e-6)
#    dxn, dhn, dyn = vector_to_weights( dn, rnn.Nx, rnn.Nh, rnn.Ny )
#    dxn2_t, dhn2_t, dyn2_t = numerical_gradient(Et, weights_to_vector(*rnn.get_weights()), 1e-6)
#    dxn2 = sum(dxn2_t)/T
#    dhn2 = sum(dhn2_t)/T
#    dyn2 = sum(dyn2_t)/T

print '--errors--'
#print 'chain-num', np.sum(np.abs(dx-dxn)), np.sum(np.abs(dh-dhn)), np.sum(np.abs(dy-dyn))
#print 'lagr-num', np.sum(np.abs(dxl-dxn)), np.sum(np.abs(dhl-dhn)), np.sum(np.abs(dyl-dyn))
print 'chain-lagr', np.sum(np.abs(dx-dxl)), np.sum(np.abs(dh-dhl)), np.sum(np.abs(dy-dyl))
print 'lagrloop-lagr', np.sum(np.abs(dxll-dxl)), np.sum(np.abs(dhll-dhl)), np.sum(np.abs(dyll-dyl))
print 'chaindoub-chain', np.sum(np.abs(dx-dx2)), np.sum(np.abs(dh-dh2)), np.sum(np.abs(dy-dy2))



        




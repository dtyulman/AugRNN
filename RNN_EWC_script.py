import numpy as np
import matplotlib.pyplot as plt
from RNN import RNN, RNN_Manager, Data, HessEWC
from script_helpers import make_data

#%% Base network
Nx=1
Nh=50
Ny=1
tau=10

try:
    base = RNN_Manager(RNN(Wx, Wh, Wy, tau, name='Base'))    
except:
    base = RNN_Manager(RNN(Nx, Nh, Ny, tau, name='Base'))
    Wx,Wh,Wy = base.rnn.get_weights()

#%%
frz = 0.99
base.rnn.freeze_weights(np.random.rand(50)<frz, freezeWh=np.random.rand(50,50)<frz, freezeWy=np.random.rand(1,50)<frz)


#%%s
T = 400
f = 0.05
trainData = Data(x = make_data('rect', T, f, 0.1),
                 y = make_data('sin', T, [f, 2*f, 3*f]))
#base.plt.plot_data(trainData.x, trainData.y, title='Training Data')

#%%
epochs=1000
base.train_network(x = trainData.x, 
                   y = trainData.y,
                   plotLoss=True,
                   epochs=epochs,
                   eta = 0.1,
                   monitorY=True)

#base.plt.plot_history(base.yHist)

#%%
base = RNN_Manager( RNN.load('50hidden_trained.pkl') )
Wx,Wh,Wy = base.rnn.get_weights()
#%%
#base.plt.plot_loss(base.loss, ax=plt.gca())
#%%

base.set_test_input(trainData.x, name='Training input 1')
base.plot_feedforward()
##base.plot_hidden()


#%%
lmbda = 0.9
try:
    base.rnn.reg = HessEWC(lmbda, base.rnn, trainData, H)
except:
    base.rnn.reg = HessEWC(lmbda, base.rnn, trainData)
    H = base.rnn.reg.H

#%%
f = 0.05
trainData2 = Data(x = -make_data('rect', T, f, 0.4),
                  y = make_data('sin', T, [f, 3*f]))
#base.plt.plot_data(trainData2.x, trainData2.y, title='Training Data 2')

#%%


epochs=1
base.train_network(x = trainData2.x, 
                   y = trainData2.y,
                   plotLoss=True,
                   epochs=epochs,
                   eta = 0.1,
                   monitorY=True)

#%%
base.set_test_input(trainData.x, name='Training input 1')
base.plot_feedforward()

base.set_test_input(trainData2.x, name='Training input 2')
base.plot_feedforward()



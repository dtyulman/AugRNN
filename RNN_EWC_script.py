import numpy as np
import copy
import matplotlib.pyplot as plt
from RNN import RNN, SubRNN, RNN_Manager, Data, L2, HessEWC
from script_helpers import make_data

##%% Base network
#Nx=1
#Nh=50
#Ny=1
#tau=10
#
#base = RNN_Manager(RNN(Wx, Wh, Wy, tau, name='Base'))
##Wx,Wh,Wy = base.rnn.get_weights() 
#T = 400
##train = Data(x = np.ones((T,Nx)),
##             y = np.sin(np.linspace(0,8*np.pi,T)).reshape(T,Ny))
#
#f = 0.05
#trainData = Data(x = make_data('rect', T, f, 0.1),
#                 y = make_data('sin', T, [f, 2*f, 3*f]))
##base.plt.plot_data(trainData.x, trainData.y, title='Training Data')
#
##%%
#epochs=1000
#base.train_network(x = trainData.x, 
#                   y = trainData.y,
#                   plotLoss=True,
#                   epochs=epochs,
#                   eta = 0.1,
#                   monitorY=True)
#
##base.plt.plot_history(base.yHist)
#
#
##%%
#
#base.set_test_input(base.train.x, name='training data')
#base.plot_feedforward()
##base.plot_hidden()
#
##longInput = make_data('rect', 10000, f, 0.1)
##base.set_test_input(longInput, name='long')
##base.plot_feedforward()

#%%
lmbda = 0.0
base.rnn.reg = HessEWC(lmbda, base.rnn, trainData)
H = base.rnn.reg.H

f = 0.1
trainData2 = Data(x = make_data('const', T),
                  y = make_data('sin', T, [f, 3*f]))
base.plt.plot_data(trainData2.x, trainData2.y, title='Training Data 2')

#%%
epochs=1000
base.train_network(x = trainData.x, 
                   y = trainData.y,
                   plotLoss=True,
                   epochs=epochs,
                   eta = 0.1,
                   monitorY=True)

base.set_test_input(trainData.x, name='training data 1')
base.plot_feedforward()

base.set_test_input(trainData2.x, name='training data 2')
base.plot_feedforward()





import numpy as np
import copy
import matplotlib.pyplot as plt
from RNN import RNN, SubRNN, RNN_Manager, Data, L2, HessEWC
from script_helpers import make_data

#%% Base network
Nx=1
Nh=50
Ny=1
tau=10

lmbda = 0.001
L2reg = L2(lmbda)

base = RNN_Manager(RNN(Wx*3, Wh*3, Wy*3, tau, reg=L2reg, name='Base'))
#Wx,Wh,Wy = base.rnn.get_weights() 
T = 400
#train = Data(x = np.ones((T,Nx)),
#             y = np.sin(np.linspace(0,8*np.pi,T)).reshape(T,Ny))

f = 0.05
train = Data(x = make_data('rect', T, f, 0.1),
             y = make_data('sin', T, [f, 2*f, 3*f]))
#base.plt.plot_data(train.x, train.y, title='Training Data')

#%%
epochs=1000
base.train_network(x = train.x, 
                   y = train.y,
                   plotLoss=True,
                   epochs=epochs,
                   eta = 0.1,
                   monitorY=True)

#base.plt.plot_history(base.yHist)


#%%

base.set_test_input(base.train.x, name='training data')
base.plot_feedforward()
#base.plot_hidden()

#longInput = make_data('rect', 10000, f, 0.1)
#base.set_test_input(longInput, name='long')
#base.plot_feedforward()

##%% Augmented network
#addNh = 50
#aug = RNN_Manager( SubRNN(Nx, Nh+addNh, Ny, tau, name='Augmented') ) #RNN with an extra node
#aug.rnn.def_subnet(hIdx=range(aug.rnn.Nh-addNh, aug.rnn.Nh), subnet=1)  #set "new" node to its own subnet
#aug.rnn.set_subnet_weights(subWh=base.rnn.Wh, subnet=0) #set "old" nodes to same weights as trained base.rnn
#
#aug.set_test_input(base.train.x, name='training data')
#aug.plot_feedforward_subnet(title='Untrained')
##aug.plot_hidden(title='Untrained')
#
#aug.set_test_input(longInput, name='long')
#aug.plot_feedforward_subnet(title='Untrained')
##%%
#aug.train_network(base.train.x, 
#                  base.train.y,
#                  epochs=epochs,
#                  monitorY=True)
#aug.plt.plot_history(aug.yHist)
#aug.plot_feedforward_subnet(title='Trained')
##aug.plot_hidden(title='Trained')
#
#aug.set_test_input(longInput, name='long')
#aug.plot_feedforward_subnet(title='Trained')
##%% Control 
#ctrl = copy.deepcopy(aug)
#ctrl.rnn.name = 'Control'
#ctrl.rnn.set_weights(*aug.Winit) #set "new" node weights same as untrained aug.rnn 
#ctrl.rnn.set_subnet_weights(*base.Winit, subnet=0) #set "old" node weights same as untrained base.rnn
#
#ctrl.set_test_input(base.train.x, name='training data')
#ctrl.plot_feedforward(title='Untrained')
##ctrl.plot_hidden(title='Untrained')
#
#ctrl.set_test_input(longInput, name='long')
#ctrl.plot_feedforward_subnet(title='Untrained')
#
##%%
#ctrl.train_network(base.train.x, 
#                   base.train.y,
#                   epochs=epochs,
#                   monitorY=True)
#ctrl.plt.plot_history(ctrl.yHist)
#ctrl.plot_feedforward(title='Trained')
##ctrl.plot_hidden(title='Trained')
#
#ctrl.set_test_input(longInput, name='long')
#ctrl.plot_feedforward_subnet(title='Trained')
##%% Compare Base, Aug, Ctrl
#fig, ax = plt.subplots()
#for net in [base, aug, ctrl]:
#    base.plt.plot_loss(net.loss, ax=ax, label='{} (Nh={})'.format(net.rnn.name, net.rnn.Nh))
#ax.legend()
#ax.set_title(ax.get_title().split('\n')[1])
#    
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


















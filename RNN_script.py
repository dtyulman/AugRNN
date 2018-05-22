import numpy as np
import copy
import matplotlib.pyplot as plt
from RNN import RNN, SubRNN, RNN_Manager 


#%% Base network
Nx=1
Nh=50
Ny=1
tau=10 
T = 400

base = RNN_Manager(RNN(Nx, Nh, Ny, tau, name='Base'))
base.train_network(x = np.ones((T,base.rnn.Nx)), 
                   y = np.sin(np.linspace(0,8*np.pi,T)).reshape(T,base.rnn.Ny),
                   plotLoss=True)
#%%
base.plt.plot_data(base.train.x, base.train.y, title='Training Data')

base.set_test_input(base.train.x, name='training data')
base.plot_feedforward()
#base.plot_hidden()

base.set_test_input(np.ones((10000,base.rnn.Nx)), name='long')
base.plot_feedforward()

#%% Augmented network
addNh = 50
aug = RNN_Manager( SubRNN(Nx, Nh+addNh, Ny, tau, name='Augmented') ) #RNN with an extra node
aug.rnn.def_subnet(hIdx=range(aug.rnn.Nh-addNh, aug.rnn.Nh), subnet=1)  #set "new" node to its own subnet
aug.rnn.set_subnet_weights(subWh=base.rnn.Wh, subnet=0) #set "old" nodes to same weights as trained base.rnn

aug.set_test_input(base.train.x, name='training data')
aug.plot_feedforward(title='Untrained')
#aug.plot_hidden(title='Untrained')

#%%
aug.train_network(base.train.x, base.train.y, plotLoss=True)
aug.plot_feedforward(title='Trained')
#aug.plot_hidden(title='Trained')

#%% Control 
ctrl = copy.deepcopy(aug)
ctrl.rnn.name = 'Control'
ctrl.rnn.set_weights(*aug.Winit) #set "new" node weights same as untrained aug.rnn 
ctrl.rnn.set_subnet_weights(*base.Winit, subnet=0) #set "old" node weights same as untrained base.rnn

ctrl.set_test_input(base.train.x, name='training data')
ctrl.plot_feedforward(title='Untrained')
#ctrl.plot_hidden(title='Untrained')
ctrl.train_network(base.train.x, base.train.y, plotLoss=True)
ctrl.plot_feedforward(title='Trained')
#ctrl.plot_hidden(title='Trained')

#%% Compare Base, Aug, Ctrl
fig, ax = plt.subplots()
for net in [base, aug, ctrl]:
    base.plt.plot_loss(net.loss, ax=ax, label='{} (Nh={})'.format(net.rnn.name, net.rnn.Nh))
ax.legend()
ax.set_title(ax.get_title().split('\n')[1])
    








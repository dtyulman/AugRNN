import numpy as np
import copy
import matplotlib.pyplot as plt
from RNN import RNN, SubRNN, RNN_Manager, Data
from script_helpers import make_data

#%% Base network
Nx=1
Nh=50
Ny=1
tau=10

T = 400
f = 0.03
fs = np.arange(1,4)*f

#base = {}
#for i in range(len(fs)):
#    base[i] = RNN_Manager(RNN(Nx, Nh, Ny, tau, name='Base{}'.format(i)))
#    base[i].train_network(x = make_data('rect', T, f, 0.2),
#                          y = make_data('sin',  T, fs[i]),
#                          eta=0.12,
#                          epochs=700*(i+1))
#    
#    fig, axs = plt.subplots(2,3)
#    base[i].plot_training_data(ax=axs.ravel()[0])
#    base[i].plt.plot_loss(base[i].loss, ax=axs.ravel()[1])
#    
#    base[i].set_test_input(base[i].train.x, name='trained input')
#    base[i].plot_feedforward(ax=axs.ravel()[2])
#    
#    base[i].set_test_input(make_data('rect', T*10, f, 0.2), name='long')
#    ax = plt.subplot2grid((2, 3), (1, 0), colspan=3)
#    base[i].plot_feedforward(ax=ax)    
#    fig.canvas.mpl_connect('resize_event', plt.tight_layout) #TODO

#%%
comb = RNN_Manager(SubRNN(Nx,Nh*3,Ny,tau, name='Combined'))
comb.rnn.def_subnet(range(  Nh,Nh*2), 1)
comb.rnn.def_subnet(range(Nh*2,Nh*3), 2)
for i in range(len(fs)):
    comb.rnn.set_subnet_weights(*base[i].rnn.get_weights(), subnet=i)
    
#comb.set_test_input(make_data('rect', T, f, 0.2), name='trained input')
#comb.plot_feedforward_subnet(title='Untrained')

comb.train_network(x=make_data('rect', T, f, 0.2),
                   y=make_data('sin', T, fs),
                   eta = 0.12,
                   epochs=700*3)

fig, axs = plt.subplots(1,2)
comb.plot_training_data(ax=axs.ravel()[0])
comb.plt.plot_loss(comb.loss, ax=axs.ravel()[1])

comb.set_test_input(comb.train.x, name='trained input')
comb.plot_feedforward_subnet(title='Trained')

comb.set_test_input(make_data('rect', T*10, f, 0.1)*2, name='long')
comb.plot_feedforward_subnet(title='Trained') 


#%%
ctrl = RNN_Manager(SubRNN(Nx,Nh*3,Ny,tau, name='Control'))
ctrl.rnn.def_subnet(range(  Nh,Nh*2), 1)
ctrl.rnn.def_subnet(range(Nh*2,Nh*3), 2)
#ctrl.rnn.set_weights(*comb.Winit)
for i in range(len(fs)):
    ctrl.rnn.set_subnet_weights(*base[i].Winit, subnet=i)

ctrl.train_network(x=make_data('rect', T, f, 0.2),
                   y=make_data('sin', T, fs),
                   eta = 0.12,
                   epochs=700*3)

fig, axs = plt.subplots(2,3)
ctrl.plot_training_data(ax=axs.ravel()[0])
ctrl.plt.plot_loss(ctrl.loss, ax=axs.ravel()[1])

ctrl.set_test_input(ctrl.train.x, name='trained input')
ctrl.plot_feedforward(ax=axs.ravel()[2])

ctrl.set_test_input(make_data('rect', T*10, f, 0.2), name='long')
ax = plt.subplot2grid((2, 3), (1, 0), colspan=3)
ctrl.plot_feedforward(ax=ax)    

#%%
fig, ax = plt.subplots()
for net in [base[0], base[1], base[2], comb, ctrl]:
    base[i].plt.plot_loss(net.loss, ax=ax, label='{} (Nh={})'.format(net.rnn.name, net.rnn.Nh))
ax.legend()
ax.set_title(ax.get_title().split('\n')[1])    
    
#%%






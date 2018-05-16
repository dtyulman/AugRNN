import numpy as np
import copy
import matplotlib.pyplot as plt
from RNN import RNN, AugRNN, RNNplotter

#%% Train 
Nx = 1
Nh = 50
Ny = 1
T = 400
tau = 10

x = np.ones((T,Nx))
yTrain = np.sin(np.linspace(0,8*np.pi,T)).reshape(T,Ny)

base = RNN(Nx,Nh,Ny,tau)
base_Winit = base.get_weights()
loss = base.sgd(x, yTrain)

#%% Test and plot
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

ax1.plot(yTrain)
ax1.plot(x)
ax1.legend(['yTrain', 'x'])
ax1.set_title('training data')

ax2.plot(loss)
ax2.set_title('loss')
ax2.set_xlabel('epoch')

#test on training input
yTest,hTest = base.feedforward(x)
ax3.plot(yTest)
ax3.plot(x)
ax3.set_title('test')
ax3.legend(['yTest', 'x'])

#test on long input
xTest = np.ones((10000,Nx))
yTest,_ = base.feedforward(xTest)
ax4.plot(yTest)
ax4.plot(xTest)
ax4.set_title('test (long)')
ax4.legend(['yTest', 'x'])


RNNplotter.plot_hidden(hTest, title='Base network')

###################################################################
#%% Add neurons to base RNN
addNh=50
aug = AugRNN.augment(base, addNh=addNh)
frz = copy.deepcopy(aug) #will need this later
aug_Winit = aug.get_weights()

description = 'Augmented, untrained'
#xTest = np.ones((5000,Nx))
#_,_,_,hTest = RNNplotter.plot_full_base_new_readout(xTest, Nh, addNh, aug, description)
#RNNplotter.plot_hidden(hTest, color=np.concatenate([np.tile('b', Nh), np.tile('r', addNh)]), title=description)
RNNplotter.plot_weights(*aug_Winit, title=description)


#%% Build a control for comparison
ctrl = AugRNN(Nx, Nh+addNh, Ny, tau)
ctrl.set_weights(*aug_Winit) #initialize weights to be the same as aug for "added" nodes
ctrl.set_subnetwork_weights(*base_Winit) #initialize weights to be the same as the init of base for "base" nodes
ctrl_Winit = ctrl.get_weights()

description =  'Control, untrained'
xTest = np.ones((5000,Nx))
_,_,_,hTest = RNNplotter.plot_full_base_new_readout(xTest, Nh, addNh, aug, description)
RNNplotter.plot_hidden(hTest, color=np.concatenate([np.tile('b', Nh), np.tile('r', addNh)]), title=description)

#%% Train augmented network control
loss_aug = aug.sgd(x, yTrain)
loss_ctrl = ctrl.sgd(x, yTrain)

fig, ax = plt.subplots()
ax.plot(loss)
ax.plot(loss_aug)
ax.plot(loss_ctrl)
ax.legend(('Base (Nh={})'.format(Nh), 'Aug (Nh={}+{})'.format(Nh, addNh), 'Ctrl (Nh={})'.format(Nh+addNh))) 
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Squared error loss')

#%%
description =  'Augmented, trained'
xTest = np.ones((5000,Nx))
_,_,_,hTest = RNNplotter.plot_full_base_new_readout(xTest, Nh, addNh, aug, description)
RNNplotter.plot_hidden(hTest, color=np.concatenate([np.tile('b', Nh), np.tile('r', addNh)]), title=description)
RNNplotter.plot_weights(*aug.get_weights(), title=description)

description =  'Control, trained'
_,_,_,hTest = RNNplotter.plot_full_base_new_readout(xTest, Nh, addNh, ctrl, description)
RNNplotter.plot_hidden(hTest, color=np.concatenate([np.tile('b', Nh), np.tile('r', addNh)]), title=description)

#################################################################
#%% Train augmented with base weights frozen
#frz.freeze_weights()
aug_Wdiff = (aug.get_weights()[0] - aug_Winit[0],
             aug.get_weights()[1] - aug_Winit[1],
             aug.get_weights()[2] - aug_Winit[2])
RNNplotter.plot_weights(*aug_Wdiff, title='Augmented, diff')


ctrl_Wdiff = (ctrl.get_weights()[0] - ctrl_Winit[0], 
              ctrl.get_weights()[1] - ctrl_Winit[1],
              ctrl.get_weights()[2] - ctrl_Winit[2])
RNNplotter.plot_weights(*ctrl_Wdiff, title='Control, diff')














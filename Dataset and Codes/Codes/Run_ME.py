#############################################################################
############################ Initialization #################################
#############################################################################

import torch
import numpy as np

load_train_data = 0
load_test_data  = 0

train_NN        = 0
train_NN_encode = 0

rsc_allocation  = 1

NN_params       = 0
params_visual   = 0
params_encode   = 0
save_NN_params  = 0

test_results    = 0

visualize       = 0

torch.manual_seed(2)

#############################################################################
#############################################################################


#############################################################################
############################## Load Data ####################################
#############################################################################

if load_train_data:
    
    from data_loader import load
    
    DATA = load(horizon = 21, num_nodes = 125, num_layers = 3, num_rsc = 7)

    X_train, Y_train = DATA.read_train(num_sample = 10000)
    
    X_train = 1 - X_train
                
                # "X" and "Y" are Nxn matrices where "N" is the number of
                # scenarios and "n" is the number of nodes.   Each row of
                # "X" is a binary vector which has a "0" when the node is
                # damaged and "1" when the node is repaired. Each element
                # of "Y" gives the time-step at which the node is repair-
                # ed and "0" if the node is not damaged.
    
    print("\nTraining data was successfully loaded!\n")
    

if load_test_data:
    
    from data_loader import load
    
    DATA = load(horizon = 21, num_nodes = 125, num_layers = 3, num_rsc = 7)
    
    X_test, Y_test = DATA.read_test(mags = [7], num_scenario = 1000)
                       
    X_test = 1 - X_test   # Now "0" implies a repaired node, "1" implies damaged node
    
    print("\nTest data was successfully loaded!\n")

#############################################################################
#############################################################################


#############################################################################
########################## Train Neural Network #############################
#############################################################################

if train_NN:
    
    from train_NN_modified import Trainer

    trainer = Trainer( X_train, Y_train, num_epoch  = 1000 , num_train     = 9000,
                                         patience   = 15   , learning_rate = 0.01,
                                         mini_batch = 32   , show_output   = 1     )
    
    model = trainer.train() # The output model contains all features and pa-
                            # rameters of the Neural Network (cf. Pytorch)
                            
    print("\nModel was successfully trained!\n")
    
    
if train_NN_encode:
    
    from train_NN_autoencoder import Trainer

    trainer = Trainer( X_train, num_epoch  = 1000 , num_train     = 35000,
                                patience   = 20   , learning_rate = 0.01,
                                mini_batch = 64   , show_output   = 1      )
    
    model = trainer.train()
                            
    print("\nModel was successfully trained!\n")

#############################################################################
#############################################################################


#############################################################################
#############################################################################


#############################################################################
######################### Resource Allocation ###############################
#############################################################################

if rsc_allocation:
    
    from data_loader import load
    from train_NN_modified import Trainer
    import matplotlib.pyplot as plt
    
    max_true    = []
    max_learned = []
    
    for i in range(2,8):
        
        print(f"\nResource No. {i} Loading...\n")
    
        DATA = load(horizon = 21, num_nodes = 125, num_layers = 3, num_rsc = i)
    
        X_train, Y_train = DATA.read_train(num_sample = 10000)
        X_train = 1 - X_train
        
        print(f"\nTraining data was successfully loaded!")
        
        X_test, Y_test = DATA.read_test(mags = [7], num_scenario = 470)                 
        X_test = 1 - X_test
        
        print(f"Test data was successfully loaded!")
        
        trainer = Trainer( X_train, Y_train, num_epoch  = 1000 , num_train     = 9500,
                                             patience   = 20   , learning_rate = 0.01,
                                             mini_batch = 32   , show_output   = 10    )
    
        model = trainer.train()
                                
        print("\nModel was successfully trained!\n")
        
        iteration = 460
        
        Y_predict = torch.round( 21*model(X_test[iteration,:]) )

        m1 = 21*torch.max(Y_test[iteration,:]).item()        
        m2 = torch.max(Y_predict).item()
        
        max_true.append(m1)
        max_learned.append(m2)
        
    interval = range(2,8)
    plt.plot(interval, max_learned, marker='o', color='r', linestyle='-', linewidth=2)
    plt.plot(interval, max_true,    marker='o', color='k', linestyle='-', linewidth=0.5)
    
    for i in range(7):
        plt.vlines(x=i+2, ymin=5, ymax=max_true[i], linestyle='--', linewidth=0.3)
    
    plt.vlines(x=2, ymin=5, ymax=max_true[0], linestyle='--', linewidth=1)
    plt.vlines(x=3, ymin=5, ymax=max_true[1], linestyle='--', linewidth=1)
    plt.vlines(x=4, ymin=5, ymax=max_true[2], linestyle='--', linewidth=1)
    plt.vlines(x=5, ymin=5, ymax=max_true[3], linestyle='--', linewidth=1)
    plt.vlines(x=6, ymin=5, ymax=max_true[4], linestyle='--', linewidth=1)
    plt.vlines(x=7, ymin=5, ymax=max_true[5], linestyle='--', linewidth=1)
    plt.vlines(x=8, ymin=5, ymax=max_true[6], linestyle='--', linewidth=1)

    plt.xlim(1.8,7.2)
    plt.ylim(5,19)
    
    plt.legend(['Predicted Recovery Time','td-INDP Recovery Time'])
    plt.xlabel('Number of Resources')
    plt.ylabel('Recovery Time')
    plt.savefig('ResourceAllocation.png', dpi=1000)
    plt.show()

#############################################################################
#############################################################################


#############################################################################
###################### Extract Neural Net Params ############################
#############################################################################

if NN_params:
    
    import matplotlib.pyplot as plt
    
    num_hidden = 1
    num_nodes  = 125
    
    W = []
    b = []
    
    for i in range(num_hidden+1):
        
        W.append( list( model.parameters() )[2*i].detach()   )
        b.append( list( model.parameters() )[2*i+1].detach() )
    
    if params_visual:
        
        interaction_matrix = np.zeros((num_nodes,num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                interaction_matrix[i,j] = torch.dot(W[0][:,i], W[1][j,:])
        
        threshold = 0.2
        
        interaction_matrix[np.abs(interaction_matrix) < threshold] = 0
        interaction_matrix = interaction_matrix.T
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(6,6)
        cax = ax.matshow(np.abs(interaction_matrix), vmin = 0, vmax = 1.1, \
                           interpolation = 'none', cmap = plt.get_cmap('PuBu'))
        fig.colorbar(cax)
        plt.vlines(x=49, ymin=0, ymax=124, color='red', linestyle='-', linewidth=3)
        plt.vlines(x=65, ymin=0, ymax=124, color='red', linestyle='-', linewidth=3)
        plt.hlines(y=49, xmin=0, xmax=124, color='red', linestyle='-', linewidth=3)
        plt.hlines(y=65, xmin=0, xmax=124, color='red', linestyle='-', linewidth=3)
        
        plt.xlabel("\nPower" + 14 * " " + "Gas" + 21 * " " + "Water")
        plt.ylabel(5 * " " + "Power" + 24 * " " + "Gas" + 20 * " " + "Water")
        
        fig.savefig('Matrix.png', dpi=1000)
        plt.show()
        
        print("\nModel parameters were successfully obtained!\n")

    
if params_encode:
    
    import numpy             as np
    import matplotlib.pyplot as plt
    
    W1 = torch.zeros((4, 3))
    W2 = torch.zeros((3, 4))
    
    W1[:, 0] = torch.sum(W[0][:, 0 : 49], 1)
    W1[:, 1] = torch.sum(W[0][:, 49: 65], 1)
    W1[:, 2] = torch.sum(W[0][:, 65:125], 1)
    
    W2[0, :] = torch.sum(W[1][0 : 49, :], 0)
    W2[1, :] = torch.sum(W[1][49: 65, :], 0)
    W2[2, :] = torch.sum(W[1][65:125, :], 0) 

    plt.subplot(2,1,1)
    plt.imshow(np.abs(W[0][:,49:65]), cmap=plt.cm.Reds)
    
    plt.subplot(2,1,2)
    WW = np.array(W[1]).T
    plt.imshow(np.abs(WW[:,49:65]), cmap=plt.cm.Reds)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    
    plt.savefig('GasBar.png', dpi=1500)
    plt.show()
        
if save_NN_params:
    
    import numpy as np
    
    for i in range(num_hidden):
        
        np.savetxt("W" + str(i+1), W[i] , delimiter = ",")
        np.savetxt("b" + str(i+1), b[i] , delimiter = ",")
        
    print("\nModel parameters were successfully saved!\n")
    
#############################################################################
#############################################################################

#############################################################################
############################## Test Output ##################################
#############################################################################

""" The test is on the repair time sequence of the test dataset """ 

if test_results:
    
    import numpy             as np
    import matplotlib.pyplot as plt
    
    plot_index = []
    
    for accuracy in range(1, 6):
        
        accuracy_index = trainer.test_data(X_test, Y_test, accuracy)
        
        a1 = int(np.mean(accuracy_index[   0:1000]))
        a2 = int(np.mean(accuracy_index[1001:2000]))
        a3 = int(np.mean(accuracy_index[2001:3000]))
        a4 = int(np.mean(accuracy_index[3001:4000]))
        
        plot_index.append([a1, a2, a3, a4])
    
    x = [6, 7, 8, 9]
        
    plt.plot(x, plot_index[0], marker = 'o')
    plt.plot(x, plot_index[1], marker = 'v')
    plt.plot(x, plot_index[2], marker = 's')
    plt.plot(x, plot_index[3], marker = '*')
    plt.plot(x, plot_index[4], marker = 'D')
    
    plt.legend(['R=1', 'R=2', 'R=3', 'R=4', 'R=5'])
    
    plt.annotate('Accuracy Radius', xy=(8.7, 98), xytext=(6.8, 38),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),)
    
    plt.vlines(x=6, ymin=20, ymax=plot_index[4][0], linestyle='--', linewidth=1)
    plt.vlines(x=7, ymin=20, ymax=plot_index[4][1], linestyle='--', linewidth=1)
    plt.vlines(x=8, ymin=20, ymax=plot_index[4][2], linestyle='--', linewidth=1)
    plt.vlines(x=9, ymin=20, ymax=plot_index[4][3], linestyle='--', linewidth=1)
    
    plt.xlim(5.9,9.1)
    plt.ylim(20,101)
    
    plt.xticks([6, 7, 8, 9])
    plt.legend(['R=1','R=2','R=3','R=4','R=5'])
    plt.xlabel('Magnitude')
    plt.ylabel('Accuracy (%)')
    plt.savefig('plot.pdf')
    plt.show()

#############################################################################
#############################################################################

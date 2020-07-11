import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping


# Modified Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-2*x))


# Shuffle the rows of data matrices
def permute_data(X, Y):
        
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    
    return X, Y


# Extract the sequence of node repairs
def extract_repair_sequence(seq):
    
    sort_seq = seq.sort()

    repair_seq_order = sort_seq[0][sort_seq[0] != 0]
    repair_seq_nodes = sort_seq[1][sort_seq[0] != 0]
    
    return repair_seq_order, repair_seq_nodes
    

# The class to impose sparse structure into the output of Neural Nets
class myLinear(nn.Linear):
    
    def __init__(self, hidden, n_output):
        
        self.main_input = None
        super(myLinear, self).__init__(hidden, n_output)

    def forward(self, input, main_input = None):
        
        self.main_input = main_input
        return self.zero_linear(input, self.weight, self.bias)

    def zero_linear(self, input, weight, bias=None):

        if input.dim() == 2 and bias is not None:
            ret = torch.addmm(bias, input, weight.t())            
        else:
            output = input.matmul(weight.t())

            if bias is not None:
                output += bias
            ret = output
        
        return ret * self.main_input
    
    
# The class that defines the Neural Net
class NeuralNetwork(nn.Module):

  def __init__(self, num_input, hidden_1, hidden_2, hidden_3, hidden_4, num_output):

      super(NeuralNetwork, self).__init__()
      
      # Define the number of perceptrons in input, output, and hidden layers
      self.num_input  = num_input
      self.hidden_1   = hidden_1
      self.hidden_2   = hidden_2
      self.hidden_3   = hidden_3
      self.hidden_4   = hidden_4
      self.num_output = num_output
      
      # Define the function that brings one layer of NN to the next
      self.linear1 = nn.Linear(self.num_input, self.hidden_1)
      self.linear2 = nn.Linear(self.hidden_1, self.hidden_2)
      self.linear3 = nn.Linear(self.hidden_2, self.hidden_3)
      self.linear4 = nn.Linear(self.hidden_3, self.hidden_4)      
      self.linear5 = myLinear(self.hidden_4,  self.num_output)
      
      self.activation = nn.LeakyReLU() # activation function of each hidden layer

  def forward(self, x):
      
      # Forward pass in the NN - Manually set in Pytorch
      output1 = self.linear1(x)
      output1 = self.activation(output1)
      
      output2 = self.linear2(output1)
      output2 = self.activation(output2)
      
      output3 = self.linear3(output2)
      output3 = self.activation(output3)
      
      output4 = self.linear4(output3)
      output4 = self.activation(output4)
      
      output5 = self.linear5(output4, x)

      return output5
  
    
# Training the NN for the infrastructure data
class Trainer:
    
    def __init__( self, X, Y , num_epoch     , num_train  , patience ,
                               learning_rate , mini_batch , show_output ):
        
        self.num_epoch     = num_epoch
        self.num_train     = num_train
        self.patience      = patience
        self.show_output   = show_output
        self.learning_rate = learning_rate
        self.mini_batch    = mini_batch
        self.X             = X
        self.Y             = Y
        self.model         = NeuralNetwork( num_input  = self.X.shape[1],
                                            hidden_1   = 100,
                                            hidden_2   = 100,
                                            hidden_3   = 100,
                                            hidden_4   = 100,
                                            num_output = self.Y.shape[1] )        
    
    def train(self):
        
        # Create list to collect loss for plot
        train_plot = []
        valid_plot = []
        
        # Collect Training Data
        X_train = self.X[0 : self.num_train, :]
        Y_train = self.Y[0 : self.num_train, :]
        
        # Collect Validationg Data
        X_valid = self.X[self.num_train + 1 : , :]
        Y_valid = self.Y[self.num_train + 1 : , :]
        
        # Choose Optimizer
        optimizer  = optim.Adam(self.model.parameters(), self.learning_rate)
        
        # Choose Loss Function
        loss_func  = nn.MSELoss()
        
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience = self.patience, verbose = True)

        for epoch in range(self.num_epoch):
            
            if epoch % 30 == 0:
                self.learning_rate = 0.1 * self.learning_rate
                optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
                print(f"\nLearning rate = {self.learning_rate}")
            
            # Shuffle training and validation data at each epoch
            X_train, Y_train = permute_data(X_train, Y_train)
            X_valid, Y_valid = permute_data(X_valid, Y_valid)
            
            # Train for each mini-batch
            for j in range( int(X_train.shape[0] / self.mini_batch) ):
                
                X_current     = X_train[j*self.mini_batch : (j+1)*self.mini_batch, :]
                Y_current     = Y_train[j*self.mini_batch : (j+1)*self.mini_batch, :]
                
                train_predict = self.model(X_current)                # Run NN
                train_loss    = loss_func(train_predict, Y_current)  # Loss
                
                X_sum = torch.sum(X_current, 0) # Find nodes which are "0" for
                                                # all of the chosen data
                
                optimizer.zero_grad() # Set gradients equal to zero
                train_loss.backward() # NN backpropagation
                
                # Indicator - make non-zero elements of "X_sum" equal to "1"
                zero_list = torch.Tensor( [0 if e == 0 else 1 for i, e in enumerate(X_sum)] )
                
                # Freeze update on first and last layers gradient 
                # updates for the non-damaged nodes
                num_hidden = 4
                
                for count, p in enumerate(self.model.parameters()):       
                    
                    if count == 0:
                        p.grad *= zero_list.reshape(1, 125)
                        
                    elif count == 2*num_hidden:
                        p.grad *= zero_list.reshape(125, 1)
                
                optimizer.step() # Pytorch perform gradient descent
                
                
            # Run the updated NN model for validation data
            valid_predict = self.model(X_valid)
            valid_loss    = loss_func(valid_predict, Y_valid)
            
            # Append loss values for plot
            train_plot.append(train_loss.item())
            valid_plot.append(valid_loss.item())
            
            # early_stopping needs the validation loss to check if it
            # has decresed, and if it does, it will make a checkpoint
            # of the current model
            early_stopping(valid_loss, self.model)
            
            if early_stopping.early_stop:                
                print("Early stopping")
                break
            
            if epoch % self.show_output == 0:                
                print( "\n# " + str(epoch+1) + " Training Loss:    " + str( float(format(train_loss, '.8f')) ) )
                print( "# "   + str(epoch+1) + " Validation Loss:  " + str( float(format(valid_loss, '.8f')) ) )
            
        plt.figure()
        plt.plot(train_plot)
        plt.plot(valid_plot)
        plt.ylim(0, 0.003)
        plt.show()
        
        return self.model
    
    
    def test_data(self, X_test, Y_test, accuracy):
        
        accuracy_index = []
        n, m = X_test.shape
        Y_predict = self.model(X_test).detach()

        for i in range(n):
            
            counter = 0
            diff    = torch.abs( 21*( Y_predict[i].reshape(-1,1) - Y_test[i, :].reshape(-1,1) ) )
            
            for j in range(m):
                
                if diff[j] <= accuracy and Y_test[i,j].item() != 0:
                    counter += 1
                    
            total = np.count_nonzero(Y_test[i, :])
            
            if total == 0:
                accuracy_index.append(100)
            else:
                accuracy_index.append(counter/total*100)
                
        print(f"Test accuracy {accuracy} was successfully done!")
            
        return accuracy_index        

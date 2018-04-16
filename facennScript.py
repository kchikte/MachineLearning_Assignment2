'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    sig = 1.0/(1.0+np.exp(-z))
    return sig

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    n = training_data.shape[0] #no. of inputs
    w1_transpose = w1.transpose()
    w2_transpose = w2.transpose()
    trainingData_bias = np.ones(shape=(training_data.shape[0],1),dtype = np.float64)
       
    training_data = np.concatenate((training_data, trainingData_bias), axis = 1) #Add bias term for input
    a = np.dot(training_data,w1_transpose)
    z = sigmoid(a)      #output from hidden layer
    
    hiddenInput_bias = np.ones(shape=(z.shape[0],1),dtype = np.float64)
    z = np.concatenate((z, hiddenInput_bias),axis=1)
    b = np.dot(z,w2_transpose)
    o = sigmoid(b)                   # output: 50000x10
    
    #1 to k encoding of training labels
    y = np.zeros(o.shape, dtype = np.float64)  #creates a vector of 50000*10
    for i in range(y.shape[0]):   
      for j in range(y.shape[1]):
        if j==training_label[i]:
          y[i][j] = 1.0             #set the class labeled value to 1 and rest to 0
      
    #Error Function (eq. 5)
    temp1 = y*np.log(o)        
    temp2 = (1.0-y)*np.log(1.0-o)
    sum = temp1 + temp2
    r=np.sum(sum)
    obj_val = (-1.0/n) * r

    #Regularized error function (eq. 15)
    temp1 = np.sum(np.sum(np.square(w1),axis=1),axis=0)
    temp2 = np.sum(np.sum(np.square(w2),axis=1),axis=0)
    temp3 =temp1 + temp2
    regularized_error = (lambdaval/(2*n))*temp3
    obj_val=obj_val+regularized_error   
   
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    #Gradient with respect to weight from hidden layer to output layer (w2)
    constant = 1/n
    grad_w2 = np.zeros(w2.shape,dtype=np.float64) #10xn
    delta_l = np.subtract(o,y) #50000x10 - 50000x10
    grad_w2=(1.0/(training_data.shape[0]))*np.dot(delta_l.transpose(),z) #10x50000 * 50000x51 
    grad_w2 = ((lambdaval*w2)/training_data.shape[0])+grad_w2
    
    #Gradient with respect to weight from input layer to hidden layer (w1)
    grad_w1=np.zeros(w1.shape,dtype=np.float64) 
    temp1 = ((1-z[:,0:n_hidden])*z[:,0:n_hidden])
    temp2 = np.dot(delta_l,w2[:,0:n_hidden])
    delta_j = temp1*temp2 
    grad_w1 = np.dot(delta_j.transpose(),training_data)
    grad_w1 = grad_w1 + (lambdaval*w1)
    grad_w1 = constant*grad_w1

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    print(obj_val)
    return (obj_val, obj_grad)
    
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    w1_transpose = w1.transpose()
    w2_transpose = w2.transpose()
   
    training_bias = np.ones(shape=(data.shape[0],1)) #all ones vector
    x = np.concatenate((data,training_bias),axis=1) #Adding bias term to input
    a = np.dot(x,w1_transpose)
    z = sigmoid(a)  #output of hidden layer
    
    hidden_bias = np.ones(shape=(z.shape[0],1))
    z = np.concatenate((z,hidden_bias),axis=1) #Adding bias term to output of hidden layer
    b = np.dot(z,w2_transpose)
    o = sigmoid(b)  #output
    labels = np.argmax(o, axis = 1)
    return labels

    
    
# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Sc Training set Accuracy:84.85308056872037%

 Validation set Accuracy:83.75234521575985%

 Test set Accuracy:84.78425435276306%ript Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

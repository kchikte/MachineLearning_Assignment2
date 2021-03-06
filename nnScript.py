import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import datetime


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sig = 1.0/(1.0+np.exp(-z))
    return sig # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    tot_data = np.vstack((train_data,validation_data,test_data))  # combining train, validation and test 
    ftr_indices=np.all(tot_data == tot_data[0,:], axis = 0)
    fltrd_data = tot_data[:,~ftr_indices] # removing columns which are similar to the first one by performing shift operations on false columns
     
    tr_len = len(train_data)
    va_len = len(validation_data)
    tst_len = len(test_data)

    train_data = fltrd_data[0:tr_len,:]  # separating train data from the filtered data
    validation_data = fltrd_data[tr_len: (tr_len + va_len),:] # separating validation data from the filtered data
    test_data = fltrd_data[(tr_len + va_len): (tr_len + va_len + tst_len),:] # separating test data from the filtered data
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

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
    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

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


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network
train_time = []
tr_acc = []
val_acc = []
tst_acc = []
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
hidden_units_set = [50,60,70,80]
n_hidden = 50
for n_units in hidden_units_set:
    # set the number of nodes in output unit
    n_class = 10
    n_hidden = n_units
    start_time=datetime.datetime.now()  # Starting timer to calculate the training time
    # initialize the weights into some random matrices

    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)
    w1 = initializeWeights(n_input, n_hidden)
    w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 8 # optimal lambda 
    #lambda_set = [0,4,8,12,16,20]  # initializing a lambda array to test regularization with various lambda values

    #for lamb in lambda_set:  # looping through recommended lambda values
    #lambdaval = lamb
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
    

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    end_time=datetime.datetime.now() # Ending timer to calculate the training time
    time_diff=end_time-start_time
    micro_sec = time_diff.seconds*1000000+time_diff.microseconds
    train_time.append(micro_sec)
    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset
    tr_acc.append(100 * np.mean((predicted_label == train_label).astype(float)))
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset
    val_acc.append(np.mean((predicted_label == validation_label).astype(float)))
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset
    tst_acc.append(100 * np.mean((predicted_label == test_label).astype(float)))
    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
    
print("Train Accuracy : "+repr(tr_acc))
print("Validation Accuracy : "+repr(val_acc))
print("Test Accuracy : "+repr(tst_acc))
print("Train Time : "+repr(train_time))
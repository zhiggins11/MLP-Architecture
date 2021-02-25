################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2021
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import math


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    meanInp = np.mean(inp)
    stdInp = np.std(inp)
    return (inp - meanInp) / stdInp


def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    new_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        new_labels[i][labels[i]] = 1
    return new_labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    Input: x is a 2d array.  Each row of the array corresponds to one example
    """
    
    expx = np.exp(x)
    expx[np.isinf(expx)] = 1000000.0 # set an upper bound
    sumExpx = np.sum(expx, axis = 1)
    return expx / sumExpx[:, None]

def compute_accuracy(outputs, targets):
    """
    Input: outputs is of shape NxM where N is number of examples and M is number of categories. otputs[i] is output of NN run on exam. i
           targets is of shape NxM where N is number of examples and M is number of categories. targets[i] is one-hot of exam. i's category
    """
    count = 0
    for i in range(len(outputs)):
        if np.argmax(outputs[i]) == np.argmax(targets[i]):
            count += 1
    return count/len(outputs)

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        Input: a is a 2d array.  Each row of a corresponds to the one sample
        """
        self.x = a
        
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, deltas):
        """
        Compute the backward pass.
        Input: deltas is a 2D array of deltas.  Each row is a delta for one example
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return np.multiply(grad, deltas) #multiply grad and deltas componentwise # return grad * deltas

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1.0 / (1.0 + np.exp(-x))


    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x)
        #return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(0.0, x)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        return np.maximum(0.1 * x, x)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        a = np.exp(-self.x)
        return a / ((1 + a)*(1 + a))

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        a = np.tanh(self.x)
        return 1 - np.multiply(a,a)

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return 1.0 * (self.x >= 0)

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        return 1.0 * (self.x >= 0) + 0.1 * (self.x < 0)
    
    def update(self, momentum, momentum_gamma, learn_rate, L2_regularization):
        """
        Placeholder so we can call update on each layer of NN
        """
        pass

    def get_weights(self):
        """
        Placeholder so we can call get_weights on each layer of NN
        """
        return None
        
    def set_weights(self, new_weights):
        """
        Placeholder so we can call set_weights on each layer of NN
        """
        pass
    
    
class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        # Input is a row vector, W is a matrix given by [in_units, out_units], b is a row vector
        np.random.seed(42)
        self.w = np.random.normal(size = (in_units, out_units))    # Declare the Weight matrix
        self.b = np.random.normal(size = out_units)    # Create a placeholder for Bias
        self.x = None #np.zeros(shape = in_units)    # Save the input to forward in this
        self.a = None #np.zeros(shape = out_units)    # Save the output of forward pass in this (without activation)

        self.d_x = None #np.zeros(shape = in_units)  # Save the gradient w.r.t x in this
        self.d_w = np.zeros(shape = (in_units, out_units))  # Save the gradient w.r.t w in this
        self.d_b = np.zeros(shape = out_units)  # Save the gradient w.r.t b in this
        
        self.previous_w_update = np.zeros(shape = (in_units, out_units)) #for using momentum in gradient descent
        self.previous_b_update = np.zeros(shape = out_units) #for using momentum in gradient descent
        
    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        # x is a 2d array, each row is the input for one example
        self.a = np.matmul(x, self.w) + self.b #this should add b to each row of xw
        self.x = x
        return np.matmul(x, self.w) + self.b #just return self.a?

    def backward(self, deltas):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        INPUT: deltas is an array of deltas from next layer.  Each row of deltas is a delta for a specific training example
        """
        # 1. update
        self.d_w = np.matmul(self.x.T, deltas)
        self.d_b = np.sum(deltas, axis = 0)
        # 2. propagate vector
        self.d_x = np.matmul(deltas, self.w.T)
        return self.d_x
    
    def update(self, momentum, momentum_gamma, learning_rate, L2_regularization):
        w_momentum, b_momentum = 0,0
        if (momentum):
            w_momentum = momentum_gamma*self.previous_w_update
            b_momentum = momentum_gamma*self.previous_b_update
        #If I understand correctly, dw, db we compute here are actually negative of gradients
        w_change = w_momentum + learning_rate*self.d_w - learning_rate*L2_regularization*self.w #might want to change to L2/#examples if we normalize
        b_change = b_momentum + learning_rate*self.d_b
        self.w = self.w + w_change
        self.b = self.b + b_change
        self.previous_w_update = w_change
        self.previous_b_update = b_change
    
    def get_weights(self):
        return self.w
        
    def get_weights_and_biases(self):
        return [self.w, self.b]
    
    def set_weights_and_baises(self, new_weights_and_biases):
        """
        Input: new_weights_and_biases is a list with nwab[0] = new weights and nwab[1] = new biases
        """
        self.w = new_weights_and_biases[0]
        self.b = new_weights_and_biases[1]
            


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.L2_penalty = config['L2_penalty']

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        # x is a [size, dimension] matrix, one row one sample
        # y is a column vector #y should also be a matrix?  each row a one-hot encoding for the corresponding sample
        self.x = x
        self.targets = targets #I think targets are one-hot encoded when reading the data?
        outputs = x
        for layer in self.layers:
            outputs = layer(outputs)
       
        self.y = softmax(outputs) #should apply softmax to each row #stored vector should be the 'category vector'
        loss1 = None
        if targets is not None:
            loss1 = self.loss(self.y, self.targets)
        return self.y, loss1

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        Input: targets is a 2D array where the ith row is the target (one-hot encoding) for ith example
                logits is a 2D array where the ith row is softmax(output_i) where output_i is the output of NN on ith example
        '''
        ###should probably be normalized by dividing by number of examples and number of categories
        logLogits = np.log(logits) # it calculates element-wise logits
        return -np.sum(logLogits * targets) + self.loss_L2_factor() # product in numpy calculates point-wise product
        # This one should be a simplier way to calculate the loss
        
    def loss_L2_factor(self):
        weights = self.get_weights()
        total = 0
        for weight in weights:
            total += np.sum(np.multiply(weight, weight))
        return self.L2_penalty*total/2

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y #delta for output layer
        for layer in reversed(self.layers):
            delta = layer.backward(delta) #each layer returns delta for previous layer
            
                
    def update(self, momentum, momentum_gamma, learning_rate, L2_penalty):
        """
        Update the weights in each layer
        """
        for layer in self.layers:
            layer.update(momentum, momentum_gamma, learning_rate, L2_penalty)
    
    def get_weights(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                weights.append(layer.get_weights())
        return weights
            
    def get_weights_and_biases(self):
        """
        Input: None
        Output: weights - a list of weights, where weights[i] is the matrix of weights for the ith layer
        """
        weights_and_biases = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                weights.append(layer.get_weights_and_biases())
        return weights
            
    def set_weights_and_biases(self, new_weights_and_biases):
        """
        Input: new_weights should be a list of weights, where new_weights[i] is the weights to set for the ith layer
        Output: None.  Weights are set for each layer
        """
        i = 0
        for layer in self.layers:
            if isinstance(layer,Layer):
                layer.set_weights_and_biases(new_weights_and_biases[i])
                i += 1


#def batchSGD(model, x_train, y_train, config):



def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    STILL NEED TO IMPLEMENT L2 REGULARIZATION
    """
    batch_size = config['batch_size'] #for readability
    previous_weights = None #for early stopping
    previous_val_loss = None #for early stopping
    for m in range(config['epochs']):
        print("epoch: " + str(m))
        #shuffling is a bit awkward, let's see if it works without it
        #shuffle = np.random.permuation(len(x_train)) #shuffle training examples for batch SGD
        
        #this takes care of early stopping
        #IMO, early stopping is awkward to implement, it may be easier to just run all the epochs and then look at the loss from each epoch.
        #if(config['early_stop'] and m%config['early_stop_epoch'] == 0):
        #    val_outputs, val_loss = model.forward(x_valid, y_valid)
        #    if (m > 0 and val_loss > previous_val_loss):
        #        print(len(previous_weights))
        #        model.set_weights(previous_weights)
        #        break
        #    previous_val_loss = val_loss
        #    previous_weights = model.get_weights()
            
        #one epoch of batch SGD
        for i in range(math.ceil(len(x_train)/batch_size)):
            #make batch
            #batch = np.array([x_train[shuffle[i]] for i in range(i*batch_size, min((i+1)*batch_size, len(x_train)))])
            #batch_targets = np.array([y_train[shuffle[i]] for i in range(i*batch_size, min((i+1)*batch_size, len(y_train)))]) 
            batch = x_train[i*batch_size:min((i+1)*batch_size, len(x_train)),:]
            batch_targets = y_train[i*batch_size:min((i+1)*batch_size, len(x_train)),:]
            
            #run forward and backwark propagation on batch and update weights
            model.forward(batch, batch_targets)
            model.backward()
            model.update(config['momentum'], config['momentum_gamma'], config['learning_rate'], config['L2_penalty'])
        
    #raise NotImplementedError("Train method not implemented")


def test(model, x_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """
    outputs, loss = model.forward(x_test, y_test)
    return compute_accuracy(outputs, y_test)

    #raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test = load_data(path="./", mode="t10k")

    # TODO: Create splits for validation data here.
    x_valid, y_valid = x_train[:6000, :], y_train[:6000, :]
    x_train, y_train = x_train[6000:, :], y_train[6000:, :]

    # TODO: train the model
    train(model, x_train, y_train, x_valid, y_valid, config)
    test_acc = test(model, x_test, y_test)

    print("Test set accuracy: " + str(test_acc))
    print("Training set accuracy: " + str(test(model, x_train, y_train)))
    #print(model.get_weights())
    # TODO: Plots
    # plt.plot(...)

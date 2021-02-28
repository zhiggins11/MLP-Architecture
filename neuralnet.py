################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2021
################################################################################

import os, gzip
import yaml
import numpy as np
import math
import matplotlib.pyplot as plt


def load_config(path):
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(data):
    #Normalizes inputs to have mean 0 and variance 1
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean)/std


def one_hot_encoding(labels, num_classes=10):
    #Returns one-hot encodings of labels
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
    one_hot_labels = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Input: x is a 2d array.  Each row of the array corresponds to one example
    Output: softmax applied to each layer of x
    """
    x[x > 500.0] = 500.0 #to prevent overflow
    x[x < -500.0] = -500.0
    expx = np.exp(x)
    expx[np.isinf(expx)] = 1000000.0  
    sumExpx = np.sum(expx, axis=1)
    return expx / sumExpx[:, None]


def compute_accuracy(outputs, targets):
    """
    Input: outputs has shape NxM where N is number of examples and M is number of categories. outputs[i] is output of NN run on exam. i
           targets has shape NxM where N is number of examples and M is number of categories. targets[i] is one-hot of exam. i's category
    """
    output_categories = np.argmax(outputs, axis = 1)
    target_categories = np.argmax(targets, axis = 1)
    return np.mean(np.equal(output_categories, target_categories))


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        return self.forward(a)

    def forward(self, a):
        """
        Input: a is a 2d array.  Each row of a corresponds to the one sample
        Output: activation function applied to each element of a
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
        Input: deltas is a 2D array of deltas.  Each row is a delta for one example
        Output: 2d array of deltas to pass to Layer above this Activation
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
            
        elif self.activation_type == 'leakyReLU':
            grad = self.grad_leakyReLU()
            
        return np.multiply(grad, deltas)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return np.maximum(0.0, x)

    def leakyReLU(self, x):
        return np.maximum(0.1 * x, x)

    def grad_sigmoid(self):
        a = np.exp(-self.x)
        return a / ((1 + a) * (1 + a))

    def grad_tanh(self):
        a = np.tanh(self.x)
        return 1 - np.multiply(a, a)

    def grad_ReLU(self):
        return 1.0 * (self.x > 0)

    def grad_leakyReLU(self):
        return 1.0 * (self.x > 0) + 0.1 * (self.x <= 0)

    def update(self, momentum, momentum_gamma, learn_rate, L2_regularization):
        #Placeholder so we can call update on each layer of NN
        pass

    #def get_weights(self):
    #    #Placeholder so we can call get_weights on each layer of NN
    #    return None

    #def set_weights(self, new_weights):
    #    #Placeholder so we can call set_weights on each layer of NN
    #    pass


class Layer:
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
        self.w = np.random.normal(size = (in_units, out_units), loc = 0.0, scale = np.sqrt(2.0 / in_units))  # Declare the Weight matrix
        self.b = np.random.normal(size=out_units, scale = np.sqrt(2.0 / in_units)) # Create a placeholder for Bias
        self.x = None  # np.zeros(shape = in_units)    # Save the input to forward in this
        self.a = None  # np.zeros(shape = out_units)    # Save the output of forward pass in this (without activation)

        self.d_x = None  # np.zeros(shape = in_units)  # Save the gradient w.r.t x in this
        self.d_w = np.zeros(shape=(in_units, out_units))  # Save the gradient w.r.t w in this
        self.d_b = np.zeros(shape=out_units)  # Save the gradient w.r.t b in this

        self.previous_w_update = np.zeros(shape=(in_units, out_units))  # for using momentum in gradient descent
        self.previous_b_update = np.zeros(shape=out_units)  # for using momentum in gradient descent

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Input: x is a matrix, each row of which is an example to run through the Layer
        Output: self.a is the result of running x through the Layer
        """
        # x is a 2d array, each row is the input for one example
        self.a = np.matmul(x, self.w) + self.b  # this should add b to each row of xw
        self.x = x
        return self.a 

    def backward(self, deltas):
        """
        INPUT: deltas is an array of deltas from next layer.  Each row of deltas is a delta for a specific training example
        Output: self.d_x is matrix of deltas to send to Activation before this Layer 
        """
        # 1. update
        self.d_w = -np.matmul(self.x.T, deltas)
        self.d_b = -np.sum(deltas, axis=0)
        # 2. propagate vector
        self.d_x = np.matmul(deltas, self.w.T)
        return self.d_x

    def update(self, momentum, momentum_gamma, learning_rate, L2_regularization):
        """
        Input:
            momentum - boolean that determines whether momentum is used in gradient descent update
            momentum_gamma - momentum factor if momentum is used
            learning_rate - learning rate for gradient descent
            L2_penalty - L2 regularization constant
        Output:
            No output. Weights and biases are updated using gradient descent update
        """
        w_momentum, b_momentum = 0, 0
        if (momentum):
            w_momentum = momentum_gamma * self.previous_w_update
            b_momentum = momentum_gamma * self.previous_b_update
        w_change = w_momentum - learning_rate * self.d_w - learning_rate * L2_regularization * self.w
        b_change = b_momentum - learning_rate * self.d_b
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
        #Creates a neural network using config
        self.layers = []
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this
        self.L2_penalty = config['L2_penalty']

        # Add layers
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Input:
            x - inputs to run through neural network.  x is a matrix of size [#examples, dimension], where each row is one example
            targets - targets for the data x.  targets is a matrix, where each row is a one-hot encoding of the category for the
                        corresponding example
        Output:
            self.y - output of running x through neural network.  self.y is a matrix of size [#examples, #categories] where each row 
                    is the output for the corresponding row of x
            loss1 - the cross entropy loss for self.y and targets.
        """
        self.x = x
        self.targets = targets
        outputs = x
        for layer in self.layers:
            outputs = layer(outputs)

        self.y = softmax(outputs)  #should apply softmax to each row of outputs
        loss1 = None
        if targets is not None:
            loss1 = self.loss(self.y, self.targets)
        return self.y, loss1

    def loss(self, logits, targets):
        '''
        Input: targets is a 2D array where the ith row is the target (one-hot encoding) for ith example
                logits is a 2D array where the ith row is softmax(output[i]) where output[i] is the output of NN on ith example
        Output: normalized cross entropy loss for logits and targets
        '''
        logLogits = np.log(logits)
        return -np.mean(logLogits * targets) + self.loss_L2_factor()

    def loss_L2_factor(self):
        """
        Computes the L2 regularization factor for the loss function using the Neuralnetwork's L2_penalty parameter
        """
        weights = self.get_weights()
        total = 0
        for weight in weights:
            total += np.sum(np.multiply(weight, weight))
        return self.L2_penalty * total / 2

    def backward(self):
        '''
        Runs backpropagation on the network to compute gradients at each layer
        '''
        delta = self.targets - self.y  # delta for output layer
        for layer in reversed(self.layers):
            delta = layer.backward(delta)  # each layer returns the delta for previous layer

    def update(self, momentum, momentum_gamma, learning_rate, L2_penalty):
        """
        Input:
            momentum - boolean that determines whether momentum is used in gradient descent update
            momentum_gamma - momentum factor if momentum is used
            learning_rate - learning rate for gradient descent
            L2_penalty - L2 regularization constant
        Output:
            No output. weights and biases of each layer are updated 
        """
        for layer in self.layers:
            layer.update(momentum, momentum_gamma, learning_rate, L2_penalty)

    def get_weights(self):
        """
        Input: None
        Output: weights - a list of weights, where weights[i] is the matrix of weights for the ith Layer
        """
        weights = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                weights.append(layer.get_weights())
        return weights

    def get_weights_and_biases(self):
        """
        Input: None
        Output: weights - a list of weights and biases, where weights[i] = [weights, biases] is the matrix of weights and
                            vector of biases for the ith Layer
        """
        weights_and_biases = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                weights_and_biases.append(layer.get_weights_and_biases())
        return weights_and_biases

    def set_weights_and_biases(self, new_weights_and_biases):
        """
        Input: new_weights should be a list of weights and biases, where new_weights[i] is [weights, biases] to set for the ith layer
        Output: None.  Weights are set for each layer
        """
        i = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.set_weights_and_baises(new_weights_and_biases[i])
                i += 1


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Input:
        model - a Neuralnetwork
        x_train - training data
        y_train - targets for training data (one-hot encoded)
        x_valid - validation data
        y_valid - targets for validation data (one-hot encoded)
        config - Configuration parameters (from config.yaml) for training
    Outputs:
        train_losses - list of losses on training set over the epochs of training
        valid_losses - list of losses on validation set over the epochs of training
        train_accuracies - list of accuracies on training set over the epochs of training
        valid_accuracies - list of accuracies on validation set over the epochs of training
    """
    print('Training the model...')
    batch_size = config['batch_size']
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    best_weights_and_biases = None
    min_loss = None
    for m in range(config['epochs']):
        print("Epoch: {}".format(m+1))
        
        #Batch SGD
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        for i in range(int(math.ceil(len(x_train) / batch_size))):
            batch = x_train[i * batch_size:min((i + 1) * batch_size, len(x_train)), :]
            batch_targets = y_train[i * batch_size:min((i + 1) * batch_size, len(x_train)), :]
            
            model.forward(batch, batch_targets)
            model.backward()
            model.update(config['momentum'], config['momentum_gamma'], config['learning_rate'], config['L2_penalty'])
        
        train_outputs, train_loss = model.forward(x_train, y_train)
        train_losses.append(train_loss)
        train_accuracies.append(compute_accuracy(train_outputs, y_train))
                        
        valid_outputs, valid_loss = model.forward(x_valid, y_valid)
        valid_losses.append(valid_loss)
        valid_accuracies.append(compute_accuracy(valid_outputs, y_valid))
        
        if (m % config['early_stop_epoch'] == 0) and (min_loss is None or valid_loss < min_loss):
            min_loss = valid_loss
            best_weights_and_biases = model.get_weights_and_biases()
        
        
    model.set_weights_and_biases(best_weights_and_biases)
    return train_losses, valid_losses, train_accuracies, valid_accuracies


def test(model, x_test, y_test):
    """
    Input:
        model - a Neuralnetwork
        x_test - Test set
        y_test - Targets for test set (one-hot encoded)
    Output:
        Accuracy of model on (x_test, y_test)
    """
    outputs, loss = model.forward(x_test, y_test)
    return compute_accuracy(outputs, y_test)

def plot_losses(train_losses, valid_losses):
    plt.title("Losses During Training")
    plt.plot(train_losses, label = "Training loss")
    plt.plot(valid_losses, label = "Validation loss")
    plt.legend(loc='best')
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.show()
    
def plot_accuracies(train_accuracies, valid_accuracies):
    plt.title("Accuracies During Training")
    plt.plot(train_accuracies, label = "Training accuracy")
    plt.plot(valid_accuracies, label = "Validation accuracy")
    plt.legend(loc='best')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    


    
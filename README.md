# Neural Network Architecture

This is work that I did as part of a group project (with Yunyi Zhang) for CSE 251B (Neural Networks).  I benefited from discussions with Yunyi, but all code here was written by me or given by the instructor.  Specifically, the instructor gave us a skeleton for the program consisting of some suggestions for classes and functions to define, and I implemented these classes and functions as well as some others needed to make the architecture.

This project consists of a class for building and training neural networks that have fully connected layers.  Currently, the only supported loss function is cross entropy loss, so this architecture should be used for classification. 

## Training a network on Fashion-MNIST

Running the file `main.py` will create a neural network with the parameters given by the file `config.yaml` (see the Parameters section below), train it on the Fashion-MNIST dataset, and compute its accuracy on the test set.

## Training a network on another dataset

To train a neural network on another dataset, simply change the block of code in `main.py` under the comment `#Load the data, shuffle the training data, and split it into training and validation sets` to load your data into the variables `x_train, y_train, x_valid, y_valid, x_test, y_test`.  Make sure to still use the `load_data` function since this will normalize the data and compute one-hot encodings for the labels.  Then run `main.py`.
    
## Parameters

Use the config.yaml to set the following parameters of the neural network

**layer_specs**: The number of layers as well as the size of each layer.  For example, layer_specs = [784, 50, 50, 10] will create a neural network with an input layer with 784 nodes, two didden layers with 50 nodes, and an output layer with 10 nodes

**activation**: The activation function to be used on each hidden layer.  Options are "tanh", "sigmoid", "ReLU", and "Leaky ReLU".

**learning_rate**: Learning rate to use for batch stochastic gradient descent

**batch_size**: Size of batches to use for stochastic gradient descent

**epochs**: Number of epochs to perform batch stochastic gradient descent for.

**early_stop_epoch**: The validation loss will be checked every early_stop_epoch number of epochs, and the network weights will be recorded if the validation loss is lower the the previous minimum validation loss.

**L2_penalty**: Loss function used for the network is E = (cross entropy loss) + L2_penalty/2 (sum w^2), where the sum is over all the weights of the network

**momentum, momentum_gamma**: If momentum is True, then the gradient descent updates will be new_weights = old_weights - (learning_rate)(gradient) + (momentum_gamma)(previous gradient descent update)




# CSE-251-Homework 2 - Neural Networks


##Parameters

Use the config.yaml to set the following parameters of the neural network

**layer_specs**: The number of layers as well as the size of each layer.  For example, layer_specs = [784, 50, 50, 10] will create a neural network with an input layer with 784 nodes, two idden layers with 50 nodes, and an output layer with 10 nodes

**activation**: The activation function to be used on each hidden layer.  Options are "tanh", "sigmoid", "ReLU", and "Leaky ReLU".

**learning_rate**: Learning rate to use for batch stochastic gradient descent

**batch_size**: Size of batches to use for stochastic gradient descent

**epochs**: Number of epochs to perform batch stochastic gradient descent for.

**early_stop, early_stop_epoch**: If early_stop is True, then the validation loss will be checked every early_stop_epoch number of epochs, and the network weights will be recorded if the validation loss goes up between two consecutive checks.

**L2_penalty**: Loss function used for the network is E = (cross entropy loss) + L2_penalty/2 (sum w^2), where the sum is over all the weights of the network

**momentum, momentum_gamma**: If momentum is True, then the gradient descent updates will be new_weights = old_weights - (learning_rate)(gradient) + (momentum_gamma)(previous gradient descent update)


##Creating and training a neural network
With the desired parameters set in the config.yaml file, config = load_config("./") will load your configuration, and model = Neuralnetwork(config) will create a neural network with your desired parameters.  Then train(model, x_train, y_train, x_valid, y_valid, config) will train your neural network using x_train as the training data with targets y_train (one-hot encodings of the classes of the training data), x_valid as the validation data with targets y_valid (one-hot encodings of the classes of the validation data), and configuration config (specified by config.yaml).

##Training and evaluating a neural network on the Fashion MNIST data set
Running the neuralnet.py file as is, our main method will perform the following steps, which are indicated using comments in the main method of the  neuralnet.py file

1. Load the config file.
2. Load and normalize the Fashion MNIST training data and test data.
3. Train the model using the given configuration.  Use 5-fold cross validation.
4. Plot the losses and accuracies.


The Sgd is implemented in the 'train' function, and the early stopping method is: to run several iterations, and store the best weights based on the performance on validation set.

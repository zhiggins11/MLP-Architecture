from neuralnet import *

if __name__ == "__main__":
    config = load_config("./")
    model = Neuralnetwork(config)

    #Load the data, shuffle the training data, and split it into training and validation sets
    x_train, y_train = load_data(path="./", mode="train")
    x_test, y_test = load_data(path="./", mode="t10k")
    
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]
    
    split = len(x_train)/10
    x_valid, y_valid = x_train[:split, :], y_train[:split, :]
    x_train, y_train = x_train[split:, :], y_train[split:, :]
    
    #Train the model
    train_losses, valid_losses, train_accuracies, valid_accuracies = train(model, x_train, y_train, x_valid, y_valid, config)
    
    plot_losses(train_losses, valid_losses)
    plot_accuracies(train_accuracies, valid_accuracies)
    
    print("Test accuracy: {} ".format(test(model, x_test, y_test)))
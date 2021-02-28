from neuralnet import *

if __name__ == "__main__":
    config = load_config("./")
    model = Neuralnetwork(config)

    x_train, y_train = load_data(path="./", mode="train")
    x_test, y_test = load_data(path="./", mode="t10k")
    
    #Shuffle the training data and split it into training and validation sets
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]

    x_valid, y_valid = x_train[:6000, :], y_train[:6000, :]
    x_train, y_train = x_train[6000:, :], y_train[6000:, :]
    
    #Train the model
    train_losses, valid_losses, train_accuracies, valid_accuracies = train(model, x_train, y_train, x_valid, y_valid, config)
    
    plot_losses(train_losses, valid_losses)
    plot_accuracies(train_accuracies, valid_accuracies)
    
    print("Test accuracy: {} ".format(test(model, x_test, y_test)))
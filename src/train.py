import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from loss import CrossEntropy
from model import Model


def train_model(X_train,
                y_train,
                X_test,
                y_test,
                input_size,
                hidden_size,
                output_size,
                activation="sigmoid",
                loss="mean_squared_error",
                learning_rate=0.01, num_epochs=1000):
    # Convert the training and test data to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Create the model
    model = Model(input_size=input_size,
                  hidden_size=hidden_size,
                  output_size=output_size,
                  activation=activation,
                  loss=loss)

    # Lists to store loss and accuracy during training and testing
    loss_values = []
    test_loss_values = []
    train_accuracy_values = []

    for epoch in range(num_epochs):
        # Forward pass
        y_pred_train = model.forward(X_train)
        y_pred_test = model.forward(X_test)  # Compute predictions for the test data

        # Backward pass
        model.backward(X_train, y_train, y_pred_train, learning_rate)

        # Compute loss and accuracy for monitoring purposes
        loss_value_train = CrossEntropy()(y_train, y_pred_train)
        loss_value_test = CrossEntropy()(y_test, y_pred_test)  # Compute test loss
        train_accuracy = np.mean(np.argmax(y_pred_train, axis=1) == np.argmax(y_train, axis=1))

        loss_values.append(loss_value_train)
        test_loss_values.append(loss_value_test)  # Store the test loss
        train_accuracy_values.append(train_accuracy)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value_train}, Test Loss: {loss_value_test}, Train Accuracy: {train_accuracy}")

    return model, loss_values, test_loss_values, train_accuracy_values


if __name__ == "__main__":
    # Load data from the CSV file
    data = pd.read_csv("../data/dataset/iris.csv")

    X = data.drop("target", axis=1).values
    y = data["target"].values

    # Normalize the features (optional but can help with convergence)
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    # Split data into training and test sets before one-hot encoding
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding for multi-class classification
    y_train_one_hot = np.zeros((y_train.shape[0], len(np.unique(y))))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1

    # Model and training parameters
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = y_train_one_hot.shape[1]  # Number of classes in the multi-class problem
    learning_rate = 0.1
    num_epochs = 1000

    # Train the model and get loss values during training and testing
    model, loss_values, test_loss_values, train_accuracy_values = train_model(X_train,
                                                                               y_train_one_hot,
                                                                               X_test,
                                                                               y_test,
                                                                               input_size,
                                                                               hidden_size,
                                                                               output_size,
                                                                               activation="sigmoid",
                                                                               loss="cross_entropy",
                                                                               learning_rate=learning_rate,
                                                                               num_epochs=num_epochs)

    # Plotting the loss values during training and testing
    plt.figure(figsize=(8, 6))
    plt.plot(range(num_epochs), loss_values, label='Train Loss')
    plt.plot(range(num_epochs), test_loss_values, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    y_pred_labels = np.argmax(predictions, axis=1)
    test_accuracy = np.mean(y_pred_labels == y_test)

    print("Test Accuracy:", test_accuracy)

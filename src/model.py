import numpy as np
from activation import Sigmoid, ReLU, Tanh
from loss import CrossEntropy, MeanSquaredError


class Model:
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 activation="sigmoid",
                 loss="cross_entropy"):
        self.hidden_output = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "relu":
            self.activation = ReLU()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            raise ValueError("Invalid activation function. Choose 'sigmoid', 'relu', or 'tanh'.")

        if loss == "cross_entropy":
            self.loss = CrossEntropy()
        elif loss == "mean_squared_error":
            self.loss = MeanSquaredError()
        else:
            raise ValueError("Invalid loss function. Choose 'cross_entropy' or 'mean_squared_error'.")

    def forward(self, X):
        self.hidden_output = self.activation(X.dot(self.W1) + self.b1)
        output = self.activation(self.hidden_output.dot(self.W2) + self.b2)  # Correction ici
        return output

    def predict(self, X):
        return self.forward(X)

    # Inside the Model class
    def backward(self, X, y_true, y_pred, learning_rate=0.01):
        m = len(y_true)

        # Compute gradients for the output layer
        d_output = self.loss.derivate(y_true, y_pred)
        d_W2 = self.hidden_output.T.dot(d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        # Propagate the gradients to the hidden layer
        d_hidden = d_output.dot(self.W2.T) * self.activation.derivate(self.hidden_output)
        d_W1 = X.T.dot(d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1

import numpy as np


class Sigmoid:
    def __call__(self, x):
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))

    def derivate(self, x):
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh:
    def __call__(self, x):
        return np.tanh(x)

    def derivate(self, x):
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivate(self, x):
        x = np.array(x)  # Convert the input to a numpy array
        return (x > 0).astype(int)


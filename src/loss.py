import numpy as np


class CrossEntropy:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        # Clip predicted values to avoid division by zero and numerical instability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calculate cross entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def derivate(self, y_true, y_pred):
        # Clip predicted values to avoid division by zero and numerical instability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calculate derivative of cross entropy loss with respect to y_pred
        d_loss = (y_pred - y_true) / ((1 - y_pred) * y_pred)
        return d_loss


class MeanSquaredError:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        # Calculate mean squared error loss
        loss = np.mean((y_true - y_pred)**2)
        return loss

    def derivate(self, y_true, y_pred):
        # Calculate derivative of mean squared error loss with respect to y_pred
        d_loss = 2 * (y_pred - y_true) / len(y_true)
        return d_loss
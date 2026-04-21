import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.W = None
    
    def _softmax(self, z):
        """
        Numerically stable softmax.
        z: shape (N, C)
        """
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """

        #bias term is handled in main.py
        
        N, D = training_data.shape
        C = get_n_classes(training_labels)

        y_onehot = label_to_onehot(training_labels, C)

        self.W = np.zeros((D, C))

        for _ in range(self.max_iters):
            scores = training_data @ self.W
            probs = self._softmax(scores)

            grad_W = (training_data.T @ (probs - y_onehot)) / N
            self.W -= self.lr * grad_W
        pred_probs = self._softmax(training_data @ self.W)
        pred_labels = onehot_to_label(pred_probs)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        scores = test_data @ self.W
        probs = self._softmax(scores)
        pred_labels = onehot_to_label(probs)

        return pred_labels

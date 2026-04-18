import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

        
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        self.mean = np.mean(training_data, axis=0)
        self.std = np.std(training_data, axis=0)
        self.std[self.std == 0] = 1
        
        self.training_data = (training_data - self.mean) / self.std
        self.training_labels = training_labels
        
        return self.predict(training_data)

        
    def predictOne(self, test_sample):
        distances = np.linalg.norm(test_sample-self.training_data, axis=1)
        indices = np.argpartition(distances,self.k)[0:self.k]
        if (self.task_kind == "classification"):
            bins = np.bincount(self.training_labels[indices].astype(int))
            return np.argmax(bins)
        else:
            return np.mean(self.training_labels[indices])

        
        
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        test_data_normalized = (test_data - self.mean) / self.std
        k_nn = [self.predictOne(sample) for sample in test_data_normalized]
        return np.array(k_nn)

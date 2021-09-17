#Usman Shoukat(201537600)
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    """Perceptron classifier.

    Parameters:
    eta:
      Learning rate
    iterations
      The number of times the data is passed to train the model

    Attributes
    -----------
    weights : 1d-array
      Weights after fitting.
    bias :
      bias after fitting

    """
    def __init__(self, eta=0.01, iterations=20):
        self.eta = eta
        self.iterations = iterations

    def fit(self, data, label):
        """Fit training data.

        Parameters:
        data :
          training data
        label :
          labels of the training data

        Returns
        self : object

        """
        #initialisation of weights and bias
        self.weights = np.zeros(data.shape[1])
        self.bias = 0
        for _ in range(self.iterations): #iterating through number of itertions
          for datapoint, y in zip(data, label): #iterating through each datapoint
              a = self.netInput(datapoint) #getting activation score through linear function
              if a * y <= 0: #checking if the prediction was wronng. If the prediction was wrong then it goes in if body
                  self.weights +=  self.eta * y * datapoint #updating the weights
                  self.bias += self.eta * y #updatig the bias 

    def netInput(self, X):
        """Calculate net input through linear equation"""
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """Return class label after chaecking it against threshold"""
        return np.where(self.netInput(X) >= 0.0, 1, -1)

    def testing(self, testData, label):
        """
        This method takes the data to be tested and its labels and returns the accuracy of the model
        """
        TP = 0
        for x, y in zip(testData, label):
            prediction = self.predict(x)
            print(x, " is predicted as :", prediction)
            if prediction == y:
                TP += 1

        return TP/testData.shape[0]

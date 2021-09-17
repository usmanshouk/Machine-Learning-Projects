from matplotlib import colors
import numpy as np
from numpy.core.fromnumeric import argmax
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Feedforward neural network / Multi-layer perceptron classifier. Uses back propagation
    to optimise the error function

    Parameters:
        iterations : Number of Epochs
        hiddenUnits : Number of neurons in the hidden layer
        eta : Learning rate
        mu : Momentum Co efficient
    """
    def __init__(self, iteration = 1000, eta= 0.01, hiddenUnits = 15, mu = 0.1):
        self.iteration = iteration
        self.eta = eta 
        self.hiddenUnits = hiddenUnits
        self.mu = mu

    def sigmoid(self, v):
        """Compute the logistic function()sigmoid"""
        return 1 / (1 + np.exp(-np.clip(v, -250, 250)))

    def forward(self, X):
        """
        This method takes a datapoint, pass it through the forward path of neural network and
        returns the induced local field and activation scores of the output layer
        """
        netInputHidden = np.dot(self.weight_h.T, X) + self.bias_h
        netOutHidden = self.sigmoid(netInputHidden)
        netInputOuterLayer = np.dot(self.weight_o.T, netOutHidden) + self.bias_o
        netOutOuterLayer = self.sigmoid(netInputOuterLayer)

        return netOutHidden, netOutOuterLayer

    def derivativeSigmoid(self, v):
        """Compute the derivative of logistic function()sigmoid"""
        return v * (1 - v)

    def fittingWithBackPropagation(self, dataset, label, validationX, validationLabel):
        """
        This method takes training and testing dataset, trains the model on the given dataset by
        using the backpropagation and plots the loss vs epoch and accuracy vs epoch curve. It also
        prints out the accuracies on the training as well as testing set
        """

        np.random.seed(100)
        #Initialisation of parameters
        self.weight_h = np.random.normal(0, 0.1, (dataset.shape[1], self.hiddenUnits))
        self.weight_o = np.random.normal(0, 0.1, (self.hiddenUnits, label.shape[1]))
        self.bias_h = np.random.normal(0, 0.1, (self.hiddenUnits))
        self.bias_o = np.random.normal(0, 0.1, (label.shape[1]))
        update_o = np.zeros(self.weight_o.shape)
        update_h = np.zeros(self.weight_h.shape)
        validationAccuracies = [] 
        trainingAccuracies = []
        trainingError = []
        validationError = []
        for _ in tqdm(range(self.iteration)): #Going through each epoch
            averageTrainingError = 0
            averageValidationError = 0
            for datapoint, d in zip(dataset, label): #Going through each datapoint

                #Going through foward path
                y_hidden, y_out = self.forward(datapoint)

                #Doing back propagation
                e = d - y_out
                averageTrainingError += 0.5 * np.sum((d - y_out) ** 2)
                deltaOut = e * self.derivativeSigmoid(y_out)
                update_o = self.eta * np.outer(y_hidden, deltaOut) + self.mu * update_o
                self.weight_o = self.weight_o + update_o
                self.bias_o = self.bias_o + self.eta * deltaOut
                deltaHidden = self.derivativeSigmoid(y_hidden) * np.sum(np.dot(self.weight_o, deltaOut))
                update_h = self.eta * np.outer(datapoint, deltaHidden) + self.mu * update_h
                self.weight_h = self.weight_h + update_h
                self.bias_h = self.bias_h + self.eta * deltaHidden


            # Going through validation data and tracking the error and accuracies
            for datapoint, d in zip(validationX, validationLabel):
                y_hidden, y_out = self.forward(datapoint)
                averageValidationError += 0.5 * np.sum((d - y_out) ** 2)
            averageTrainingError /= dataset.shape[0]
            averageValidationError /= validationX.shape[0]


            validationError.append(averageValidationError)
            trainingError.append(averageTrainingError)
            

            trainingAccuracies.append(accuracy(self, dataset, label))
            validationAccuracies.append(accuracy(self, validationX, validationLabel))
        
        #Plotting the accuracy vs epoch and accuracy vs loss curves
        self.plot(trainingAccuracies, validationAccuracies, trainingError, validationError)

    def plot(self, trainigAccuracy, validationAccuracy, trainingError, validationError):
        """
        This method takes the error function and accuracy of training and validation set and
        plot their curves
        """
        
        movingAverage = 10
        smoothenedTrainigAccuracy = []
        smoothenedValidationAccuracy = []
        smoothenedTrainigError = []
        smoothenedValidationError = []

        #Smoothening the loss and accuracy curves over the moving average of 10
        for i in range(0, self.iteration, movingAverage):
            smoothenedTrainigAccuracy.append(sum(trainigAccuracy[i:i+movingAverage])/movingAverage)
            smoothenedValidationAccuracy.append(sum(validationAccuracy[i:i+movingAverage])/movingAverage)
            smoothenedTrainigError.append(sum(trainingError[i:i+movingAverage])/movingAverage)
            smoothenedValidationError.append(sum(validationError[i:i+movingAverage])/movingAverage)
        f, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(range(0, self.iteration, movingAverage), smoothenedTrainigError, color = "r", label = "Training")
        ax1.plot(range(0, self.iteration, movingAverage), smoothenedValidationError, color = "g", label = "Validation")
        ax1.legend()
        ax1.set_title("Error VS Iterations")
        ax1.set_ylabel("Error Function")
        ax2.plot(range(0, self.iteration, movingAverage), smoothenedTrainigAccuracy, color = "r", label = "Training")
        ax2.plot(range(0, self.iteration, movingAverage), smoothenedValidationAccuracy, color = "g", label = "Validation")
        ax2.set_title("Accuracy VS Iterations")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Accuracy(%)")
        ax2.legend()
        plt.suptitle('Hidden Units "' + str(self.hiddenUnits) + '"' + ', Learning rate "' + str(round(self.eta, 2)) + '"' + ' and Momentum coeficient "' + str(round(self.mu, 2)) + '"')
        plt.show()

def takeHyperParameters():
    '''This function takes the hyperparameters from the user and return these hyoer parameters'''

    print("\nEnter the following hyperparameters\n")
    iterations = int(input("Enter the number of Epochs for which you want to train the model : "))
    eta = float(input("Enter the value of learning rate : "))
    hiddenUnits = int(input("Enter the number of hidden units : "))
    mu = float(input("Enter the momentum coefficient : "))
    return iterations, eta, hiddenUnits, mu

                
def accuracy(nn, X, y):
    """
        This method takes the dataset and runs it through the forward path of the network, gets the
        activation score of the output layer, check it against the given label and give a prediction
        about which class it belongs to. By taking the account of true predictions, at the end it returns
        the accuracy of the model on the dataset
    """
    tp = 0
    for data, d in zip(X, y):
        a,b = nn.forward(data)
        if argmax(b) == argmax(d):
            tp += 1
        accuracy = round(tp/y.shape[0] * 100, 2)
    return accuracy

def crab():
    """
    This method loads the crab dataset and train a neural network model the dataset
    """
    #Loading the crab dataset
    with open("crabData.csv") as file:
        data = []
        for line in file:
            data.append(line.strip().split(","))
        file.close()

    np.random.seed(4738)
    #Cleaning and splitting the data in training and testing data
    data = np.array(data)
    data = data[1:,:]
    np.random.shuffle(data)
    dataDivisior = int(data.shape[0]*0.8)
    X = data[:dataDivisior, 3:].astype(np.float)
    validationX = data[dataDivisior:, 3:].astype(np.float)
    #One hot encoding the labels
    Y = np.array(pd.get_dummies(data[:, 1]))
    trainingLabel = Y[:dataDivisior,:]
    validationLabel = Y[dataDivisior:,:]
    iterations, eta, hiddenUnits, mu = takeHyperParameters()

    #initialising and training the network
    network = NeuralNetwork(iteration = iterations,hiddenUnits=hiddenUnits, eta= eta, mu = mu)
    network.fittingWithBackPropagation(X, trainingLabel, validationX, validationLabel)
    print("Training Accuracy : " + str(accuracy(network, X, trainingLabel)) + "%")
    print("Testing Accuracy : " + str(accuracy(network, validationX, validationLabel)) + "%")


def Iris():
    """
    This method loads the Iris dataset and train a neural network model the dataset
    """
    #Loading the Iris Dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)

    np.random.seed(1210)
    #Cleaning and splitting the data in training and testing data
    permutaion = np.random.permutation(150)
    df = df.iloc[permutaion]
    #One hot encoding the labels
    Y = np.array(pd.get_dummies(df[4]))
    data = np.array(df)
    data = data[:,:4]
    dataDivisior = int(data.shape[0]*0.8)
    X = data[:dataDivisior,:].astype(np.float)
    trainingLabel = Y[:dataDivisior,:]
    validationLabel = Y[dataDivisior:,:]
    validationX = data[dataDivisior:, :].astype(np.float)
    iterations, eta, hiddenUnits, mu = takeHyperParameters()

    #initialising and training the network
    network = NeuralNetwork(iteration = iterations,hiddenUnits=hiddenUnits, eta= eta, mu = mu)
    network.fittingWithBackPropagation(X, trainingLabel, validationX, validationLabel)
    print("Training Accuracy : " + str(accuracy(network, X, trainingLabel)) + "%")
    print("Testing Accuracy : " + str(accuracy(network, validationX, validationLabel)) + "%")


def main():
    """This method ask the user about the dataset on which you want to train your model and direct
    the program to that dataset"""
    
    print("\n\t\t\tWelcome")
    while True:
        print("\nChoose from the following on which dataset you want to train your model")
        menu = input("1. Press 1 for Iris dataset(Classification between Setosa, Versicular and Virginka)\n2. Press 2 for Crab dataset(Classification between Male and Female)1\n3. Press Q/q to quit : ")
        if menu == "1":
            Iris()
        elif menu == "2":
            crab()
        elif menu == "Q" or menu == "q":
            break
        else:
            print("\nEnter Valid Entry\n")


if __name__ == "__main__":
    main()


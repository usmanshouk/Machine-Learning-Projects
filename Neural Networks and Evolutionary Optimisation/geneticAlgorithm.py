import numpy as np
from numpy.core.fromnumeric import argmax
from geneticalgorithm import geneticalgorithm as ga
import pandas as pd

class NeuralNetwork:
    """
    Feedforward neural network / Multi-layer perceptron classifier. Uses Genetic Algorithm
    to optimise the error function

    Parameters:
        iterations : Number of Epochs
        hiddenUnits : Number of neurons in the hidden layer
        populationSize : Number of individuals in the population
        mutationProbability : Mutation rate
        eliteRatio : Proportion of elites in the population
        crossoverProbability : Crossover Rate
        parentsPortion : The portion of population filled by the members of the previous generation 
        crossoverType : Type of crossover (uniform, one_point, two_point)
        variableType : Type of variable (Real or Bool)
        lb : Lower bound of the variable
        ub : upper bound of the variable
    """
    def __init__(self, hiddenUnits = 32, iterations  = 100, populationSize = 150, mutationProbability = 0.2, elitRatio = 0.1, crossoverProbability = 0.5, parentsPortion = 0.2, crossoverType = "uniform", variableType = "real", lb = -15, ub = 15):
        self.hiddenUnits = hiddenUnits
        self.iterations = iterations
        self.populationSize = populationSize
        self.mutationProbability = mutationProbability
        self.elitRatio = elitRatio
        self.crossoverProbability = crossoverProbability
        self.parentsPortion = parentsPortion
        self.crossoverType = crossoverType
        self.variableType = variableType
        self.lb = lb
        self.ub = ub

    def forward(self, X):
        """
        This method takes a datapoint, pass it through the forward path of neural network and
        returns the activation scores of the output layer
        """
        
        netInputHidden = np.dot(X, self.w_hidden) + self.b_hidden
        netOutHidden = self.sigmoid(netInputHidden)
        netInputOuterLayer = np.dot(netOutHidden, self.w_out) + self.b_out
        netOutOuterLayer = self.sigmoid(netInputOuterLayer)

        return netOutOuterLayer

    def sigmoid(self, v):
        """Compute the logistic function()sigmoid"""
        return 1 / (1 + np.exp(-np.clip(v, -250, 250)))

    def errorFunction(self,X):
        """
        This method is the error function that we want to otimise through GA
        """

        #Splitting falttened vector of parameters in respective sets of matrix(weights) and vector form(bias)
        self.w_hidden = X[:int(self.dataset.shape[1] * self.hiddenUnits)].copy().reshape((self.dataset.shape[1],self.hiddenUnits))
        self.b_hidden = X[int(self.dataset.shape[1] * self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits)].copy()
        self.w_out = X[int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1]))].copy().reshape((self.hiddenUnits,self.label.shape[1]))
        self.b_out = X[int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1])):].copy()

        netOutOuterLayer = self.forward(self.dataset)
        #mean squared error
        e = np.sum(0.5*(self.label - netOutOuterLayer)**2)

        return e

    def fittingWithGA(self, dataset, label):
        """
        This method takes training and testing dataset, trains the model on the given dataset by
        using the GA and plots the loss vs epoch curve. 
        """

        self.dataset = dataset
        self.label = label
        #Getting the dimensions to Flatten the matrix of weights and biases to a single  vector 
        d = (dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits * label.shape[1]) + self.label.shape[1]
        #Passing the hyperparameters
        boundary = np.c_[np.zeros(d) + self.lb, np.zeros(d)+ self. ub]
        algorithm_param = {'max_num_iteration': self.iterations,\
            'population_size':self.populationSize,\
                'mutation_probability':self.mutationProbability,\
            'elit_ratio':self.elitRatio,\
                'crossover_probability': self.crossoverProbability,\
                'parents_portion': self.parentsPortion,\
                'crossover_type':self.crossoverType,\
                'max_iteration_without_improv':None}
        #calling instance of GA
        model= ga(function=self.errorFunction, dimension=d,variable_boundaries= boundary , variable_type=self.variableType, algorithm_parameters=algorithm_param)
        model.run()
        solution=model.output_dict["variable"]
        #converting the optimised flattened vector back to matrix and vector form(for weights and biases respectively)
        self.w_hidden = solution[:int(self.dataset.shape[1] * self.hiddenUnits)].copy().reshape((self.dataset.shape[1],self.hiddenUnits))
        self.b_hidden = solution[int(self.dataset.shape[1] * self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits)].copy()
        self.w_out = solution[int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1]))].copy().reshape((self.hiddenUnits,self.label.shape[1]))
        self.b_out = solution[int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1])):].copy()

    def evaluation(self,dataset, label, accuracyType):
        """
        This method takes the dataset and runs it through the forward path of the network, gets the
        activation score of the output layer, check it against the given label and give a prediction
        about which class it belongs to. Taking the account of true predictions, at the end it prints
        out the accuracy of the model on the dataset
        """
        Tp = 0
        for data, d in zip(dataset, label):
            b = self.forward(data)
            if argmax(b) == argmax(d):
                Tp += 1
        print(accuracyType + " : " + str(round(Tp/label.shape[0] * 100, 2)) + "%")
                

def takeHyperParameters():
    '''This function takes the hyperparameters from the user and return these hyoer parameters'''

    print("\nEnter the following hyperparameters\n")
    hiddenUnits = int(input("Enter the number of hidden units : "))
    iterations = int(input("Enter the number of Epochs for which you want to train the model : "))
    populationSize = int(input("Enter the Population Size : "))
    mutationProbability = float(input("Enter the mutation probability : "))
    crossoverProbability = float(input("Enter the Crossover Probability : "))
    crossoverType = input("Enter the crossover type(uniform, one_point, two_point) : ")
    variableType = input("Enter the variable type(real or bool) : ")
    lb = float(input("Enter the lower bound of the variable : "))
    ub = float(input("Enter the upper bound of the variable: "))
    
    return iterations, populationSize, hiddenUnits, mutationProbability, crossoverProbability, crossoverType, variableType, lb, ub
                
def crab():
    #Loading the crab dataset
    with open("crabData.csv") as file:
        data = []
        for line in file:
            data.append(line.strip().split(","))
        file.close()

    np.random.seed(5436)
    #Cleaning and splitting the data in training and testing data
    data = np.array(data)
    data = data[1:,:]
    np.random.shuffle(data)
    dataDivisior = int(data.shape[0]*0.8)
    X = data[:dataDivisior, 3:].astype(np.float)
    testX = data[dataDivisior:, 3:].astype(np.float)
    #One hot encoding the labels
    Y = np.array(pd.get_dummies(data[:, 1]))
    trainingLabel = Y[:dataDivisior,:]
    testingLabel = Y[dataDivisior:,:]
    iterations, populationSize, hiddenUnits, mutationProbability, crossoverProbability, crossoverType, variableType, lb, ub = takeHyperParameters()
    #initialising and training the network
    network = NeuralNetwork(iterations = iterations, populationSize = populationSize, hiddenUnits = hiddenUnits, mutationProbability = mutationProbability, crossoverProbability = crossoverProbability, crossoverType = crossoverType, variableType = variableType, lb =lb, ub = ub)
    network.fittingWithGA(X, trainingLabel)

    #evaluating the model on training and testing data
    network.evaluation(X,trainingLabel, "Traniing Accuracy")
    network.evaluation(testX, testingLabel, "Testing Accuracy")

def Iris():
    #Loading the Iris dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)
    np.random.seed(100)
    permutaion = np.random.permutation(150)
    df = df.iloc[permutaion]
    #One hot encoding the labels
    Y = np.array(pd.get_dummies(df[4]))
    data = np.array(df)
    data = data[:,:4]
    #Cleaning and splitting the data in training and testing data
    dataDivisior = int(data.shape[0]*0.8)
    X = data[:dataDivisior,:].astype(np.float)
    trainingLabel = Y[:dataDivisior,:]
    testingLabel = Y[dataDivisior:,:]
    testX = data[dataDivisior:, :].astype(np.float)
    iterations, populationSize, hiddenUnits, mutationProbability, crossoverProbability, crossoverType, variableType, lb, ub = takeHyperParameters()
    #initialising and training the network
    network = NeuralNetwork(iterations = iterations, populationSize = populationSize, hiddenUnits = hiddenUnits, mutationProbability = mutationProbability, crossoverProbability = crossoverProbability, crossoverType = crossoverType, variableType = variableType, lb =lb, ub = ub)
    network.fittingWithGA(X, trainingLabel)

    #evaluating the model on training and testing data
    network.evaluation(X,trainingLabel, "Training Accuracy")
    network.evaluation(testX, testingLabel, "Testing Accuracy")

def main():
    #Asking the user on which dayaset, you want to train your model
    print("\n\t\t\tWelcome")
    while True:
        print("\nChoose from the following on which dataset you want to train your model")
        menu = input("1. Press 1 for Iris dataset(Classification between Setosa, Versicular and Virginka)\n2. Press 2 for Crab dataset(Classification between Male and Female)\n3. Press Q/q to quit : ")
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
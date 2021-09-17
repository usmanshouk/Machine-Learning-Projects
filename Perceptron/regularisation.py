#Usman Shoukat(201537600)
#Code might will through runtime error, to avoid this use the following command: python -W ignore regularisation.py

import numpy as np
from numpy.lib.function_base import select
from math import exp
class Perceptron:
    """Perceptron classifier.

        Parameters:
        lamda:
        regularisation coefficient
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
    def __init__(self, lamda = 0.1, eta = 1, iterations=20):
        self.iterations = iterations
        self.eta = eta
        self.lamda = lamda

    def fit(self, data, classLabel):
        """Fit training data.

        Parameters:
        data :
          training data
        classlabel :
          labels of the training data

        Returns
        self : object

        """
        #Labeling the data
        data[:,4] = np.where(data[:,4] == classLabel, 1,-1)
        #shuffling the data
        np.random.shuffle(data)
        #converting the data to float values
        X = data[:,0:4].astype(np.float)
        label = data[:,4].astype(np.float)
        #initialising of weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.iterations): #iterating through number of itertions
            for datapoint, y in zip(X, label): #iterating through each datapoint
                a = self.netInput(datapoint) #getting activation score through linear function
                if a * y <= 0: #checking if the prediction was wronng. If the prediction was wrong then it goes in if body
                    self.weights = ((1-(2* self.lamda)) * self.weights) + (self.eta * y * datapoint) #updating the weights
                    self.bias = self.bias + (self.eta * y) #updatig the bias

    def netInput(self, datapoint):
        """Calculate net input through linear equation"""
        return np.dot(self.weights, datapoint) + self.bias


    
def evaluation(dataSet,pp1,pp2,pp3):
    """This method takes the test data, and the models, labels the test data, check it against the
    model and returns the accuracy of model
    Parameters:
    dataset:
        Testing Data
    pp1, pp2, pp3:
        Three trained models

    return:
        accuracy of each class, accuracy of model
    """
    #Labeling the data
    dataSet[:,4] = np.where(dataSet[:,4] == "class-1", 1, dataSet[:,4])
    dataSet[:,4] = np.where(dataSet[:,4] == "class-2", 2, dataSet[:,4])
    dataSet[:,4] = np.where(dataSet[:,4] == "class-3", 3, dataSet[:,4])
    dataSet = dataSet.astype(np.float)

    TP = 0
    classAccuracy = np.zeros(3)
    for datapoint in dataSet:
        #Getting score from each model
        score = []
        score.append(pp1.netInput(datapoint[:4]))
        score.append(pp2.netInput(datapoint[:4]))
        score.append(pp3.netInput(datapoint[:4]))
        #Getting the prediction from the model giving the highest score
        prediction = (np.argmax(score) + 1)
        #Checking for correct prediction
        if prediction == datapoint[4]:
            #Updating the correct prediction variables
            classAccuracy[int(datapoint[4])-1] += 1
            TP += 1
    accuracy = round(TP/ dataSet.shape[0], 4) * 100
    return classAccuracy/(dataSet.shape[0]/3)*100, accuracy

def dataLoader(file):
    """This methos takes the csv file return the data in the form of nd array"""
    data = []
    with open(file) as file:
        for line in file:
            data.append(line.strip().split(","))
        file.close()
    return np.array(data)
def main():
    np.random.seed(100)
    #Loading the data
    trainData = dataLoader("train.data")

    testData = dataLoader("test.data")


    print("\n\t\t\tAccuracies for Multiclassification Classification")
    print("\nAccuracy Type\t\tlambda\t\tClass1\t\tClass2\t\tClass3\t\tOverall\n")

    lamda = 0.01
    while lamda <= 100: #iterating for given values of lambda

        #Initialisation and fitting of models
        pp1 = Perceptron(lamda)
        pp1.fit(trainData.copy(), "class-1")
        pp2 = Perceptron(lamda)
        pp2.fit(trainData.copy(), "class-2")
        pp3 = Perceptron(lamda)
        pp3.fit(trainData.copy(), "class-3")



        #Calling the evaluation function to check the accuracy of given model
        classesTrainingAccuracy, overallTrainingAccuracy = evaluation(trainData.copy(),pp1,pp2,pp3)
        classesTestingAccuracy, overallTestingAccuracy = evaluation(testData.copy(),pp1,pp2,pp3)

        print("Training\t\t", lamda, "\t\t", str(classesTrainingAccuracy[0]) + "%\t\t" + str(classesTrainingAccuracy[1]) + "%\t\t" + str(classesTrainingAccuracy[2])+ "%\t\t" + str(overallTrainingAccuracy)+ "%")
        print("Testing \t\t", lamda, "\t\t", str(classesTestingAccuracy[0]) + "%\t\t" + str(classesTestingAccuracy[1]) + "%\t\t" + str(classesTestingAccuracy[2])+ "%\t\t" + str(overallTestingAccuracy)+ "%")
        lamda *= 10

    print("\n\n")

if __name__ == "__main__":
    main()

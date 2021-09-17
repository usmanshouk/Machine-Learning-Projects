import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import pyswarms as ps
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    
    
    """
    def __init__(self, iterations = 100, hiddenUnits = 32, particles =  1295, c1 = 0.5, c2 = 0.7, w = 0.9, lb = -15, ub = 15):
        self.hiddenUnits = hiddenUnits
        self.iterations = iterations
        self.particles = particles
        self.c1 = c1
        self.c2 = c2
        self.w = w


    def sigmoid(self, v):
        return 1 / (1 + np.exp(-np.clip(v, -250, 250)))

    def forward(self, X):
        netInputHidden = np.dot(self.weight_h.T, X) + self.bias_h
        netOutHidden = self.sigmoid(netInputHidden)
        netInputOuterLayer = np.dot(self.weight_o.T, netOutHidden) + self.bias_o
        netOutOuterLayer = self.sigmoid(netInputOuterLayer)

        return netOutOuterLayer

    def errorFunction(self,X):
      particles = []
      for i in range(X.shape[0]):

        w_hidden = X[i,:int(self.dataset.shape[1] * self.hiddenUnits)].copy().reshape((self.dataset.shape[1],self.hiddenUnits))
        b_hidden = X[i,int(self.dataset.shape[1] * self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits)].copy()
        w_out = X[i,int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1]))].copy().reshape((self.hiddenUnits,self.label.shape[1]))
        b_out = X[i,int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1])):].copy()

        netInputHidden = np.dot(self.dataset, w_hidden) + b_hidden
        netOutHidden = self.sigmoid(netInputHidden)
        netInputOuterLayer = np.dot(netOutHidden, w_out) + b_out
        netOutOuterLayer = self.sigmoid(netInputOuterLayer)

        e = np.sum(0.5*(self.label - netOutOuterLayer)**2)
        particles.append(e)

      return particles
    def evaluation(self,dataset, label, accuracyType):
        Tp = 0
        for data, d in zip(dataset, label):
            b = self.forward(data)
            if argmax(b) == argmax(d):
                Tp += 1
        print(accuracyType +" : " + str(round(Tp/label.shape[0] * 100, 2)) + "%")


    def fittingWithPSO(self, dataset, label, testX, testingLabel):

                self.dataset = dataset
                self.label = label
                

                d = (dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits * label.shape[1]) + self.label.shape[1]


                # Set-up hyperparameters
                options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}
                

                # Call instance of PSO
                optimizer = ps.single.GlobalBestPSO(n_particles=self.particles, dimensions=d, options=options)
                cost, pos = optimizer.optimize(self.errorFunction, iters=self.iterations)

                self.weight_h = pos[:int(self.dataset.shape[1] * self.hiddenUnits)].copy().reshape((self.dataset.shape[1],self.hiddenUnits))
                self.bias_h = pos[int(self.dataset.shape[1] * self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits)].copy()
                self.weight_o = pos[int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits):int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1]))].copy().reshape((self.hiddenUnits,self.label.shape[1]))
                self.bias_o = pos[int((self.dataset.shape[1] * self.hiddenUnits) + self.hiddenUnits + (self.hiddenUnits*self.label.shape[1])):].copy()

                plt.plot(optimizer.cost_history)
                plt.xlabel("Epochs")
                plt.ylabel("Error Function")
                plt.title("c1="+str(self.c1)+", c2="+str(self.c2)+", w="+str(self.w)+", particles="+str(self.particles))
                plt.show()


                self.evaluation(self.dataset,self.label, "Training Accuracy")
                self.evaluation(testX, testingLabel, "Testing Accuracy")
                
def takeHyperParameters():

    print("\nEnter the following hyperparameters\n")
    hiddenUnits = int(input("Enter the number of hidden units : "))
    iterations = int(input("Enter the number of Epochs for which you want to train the model : "))
    particles = int(input("Enter the number of particles : "))
    c1 = float(input("Enter the conginitive component coefficient c1 : "))
    c2 = float(input("Enter the social component coefficient c2 : "))
    w = float(input("Enter the Intertia weight w : "))

    
    return hiddenUnits, iterations, particles, c1, c2, w
                


def crab():
    with open("data.csv") as file:
        data = []
        for line in file:
            data.append(line.strip().split(","))
        file.close()

    np.random.seed(5436)
    data = np.array(data)
    data = data[1:,:]
    np.random.shuffle(data)
    dataDivisior = int(data.shape[0]*0.8)
    X = data[:dataDivisior, 3:].astype(np.float)
    testX = data[dataDivisior:, 3:].astype(np.float)
    Y = np.array(pd.get_dummies(data[:, 1]))
    trainingLabel = Y[:dataDivisior,:]
    testingLabel = Y[dataDivisior:,:]

    hiddenUnits, iterations, c1, c2, w = takeHyperParameters()
    network = NeuralNetwork(hiddenUnits = hiddenUnits, iterations = iterations, particles = particles, c1 = c1, c2 = c2, w = w)
    network.fittingWithPSO(X, trainingLabel, testX, testingLabel)

def Iris():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)
    np.random.seed(100)
    permutaion = np.random.permutation(150)
    df = df.iloc[permutaion]
    Y = np.array(pd.get_dummies(df[4]))
    data = np.array(df)
    data = data[:,:4]
    dataDivisior = int(data.shape[0]*0.8)
    X = data[:dataDivisior,:].astype(np.float)
    trainingLabel = Y[:dataDivisior,:]
    testingLabel = Y[dataDivisior:,:]
    testX = data[dataDivisior:, :].astype(np.float)
    hiddenUnits, iterations, particles, c1, c2, w = takeHyperParameters()
    network = NeuralNetwork(hiddenUnits = hiddenUnits, iterations = iterations, particles = particles, c1 = c1, c2 = c2, w = w)
    network.fittingWithPSO(X, trainingLabel, testX, testingLabel)

def main():
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
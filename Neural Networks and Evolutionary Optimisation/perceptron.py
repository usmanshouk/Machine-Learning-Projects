from os import error
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
import pandas as pd
class Perceptron:
    """Perceptron classifier.

    Parameters:
    eta:
      Learning rate
    epoch
      The number of times the data is passed to train the model

    Attributes
    -----------
    weights : 1d-array
      Weights after fitting.
    bias :
      bias after fitting
    """
    def __init__(self, eta, epoch):
        self.eta = eta
        self.epoch = epoch

    def train(self, dataset, showLearning = False):
        """Fit training data.

        Parameters:
        dataset :
          training data
        showLearning :
          If True it plots the learned boundries after each epoch, other skip it

        Returns
        self : object

        """

        X = dataset[:,:-1]
        label = dataset[:,-1]
        #Initialising the parameters
        self.weights = np.random.rand(X.shape[1])
        self.bias = np.random.rand()
        #Dictionary to store learned parameters for each epoch
        self.showLearning = {}
        
        for _ in range(self.epoch):#Going for each epoch
            errors = 0
            #This part is just to visualise the boundries while learning
            if showLearning == True:
                if X.shape[1] == 2:
                    x = np.linspace(-100,100,10)
                    y = (x * self.weights[0] + self.bias)/(-1*self.weights[1])
                    self.showLearning[_] = (x, y)
                elif X.shape[1] == 3:
                    x = np.linspace(-5,20,100)
                    y = np.linspace(-5,20,100)
                    x , y = np.meshgrid(x,y)
                    z = (-self.weights[0] * x - self.weights[1] * y - self.bias)/self.weights[2]
                    self.showLearning[_] = (x, y, z)
                else:
                    assert "Dimension are more than three and learning is True"
            #Going for each datapoint
            for datapoint, y in zip(X, label):
                a = self.predict(datapoint)
                update = self.eta * (y-a)
                #updating parameters if there is misclassification
                self.weights = self.weights + update * datapoint
                self.bias += update
                errors += int(update != 0.0)
            print("Epoch : ", _ + 1)
            print("Weights : ", self.weights)
            print("Bias : ", self.bias)


            
            if errors == 0: #Stopping the iteration, if there is no mis classification for one epoch
                break

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def selfCreated2D():

    #Creating the 2D dataset
    n = int(input("Enter the number of datapoints for which you want to generate the data : "))
    # np.random.seed(10)
    x = np.random.multivariate_normal((2,2), [[1,0],[0,1]], int(n/2))
    x1 = np.c_[x,np.ones(int(n/2))]
    y = np.random.multivariate_normal((10,5), [[1,0],[0,1]], int(n/2))
    y1 = np.c_[y,np.ones(int(n/2))*-1]
    whole_dataset = np.r_[x1,y1]
    np.random.shuffle(whole_dataset)

    
    while True:
        showLearning = input("Press 1 if you want to visualise the learning process otherwise press 0 : ")
        if showLearning == "1":
            showLearning = True
            break
        elif showLearning == "0":
            showLearning = False
            break
        else:
            print("Enter correct input : ")

    #creating theb perceptron instance and training it
    pp1 = Perceptron(0.01, 100)
    pp1.train(whole_dataset, showLearning = showLearning)

    #Plotting the boundry during the learning
    if showLearning == True:
        for k in range(len(pp1.showLearning)):

            plt.scatter(x[:,0], x[:,1], color = "r", marker = "x")
            plt.scatter(y[:,0], y[:,1], color = "b", marker = "o")
            plt.plot(pp1.showLearning[k][0], pp1.showLearning[k][1])
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Training")
            plt.xlim(-5,15)
            plt.ylim(-5,10)
            plt.show()
    else: 
        x1 = np.linspace(-100,100,10)
        y1 = (x1 * pp1.weights[0] + pp1.bias)/(-1*pp1.weights[1])
        plt.scatter(x[:,0], x[:,1], color = "r", marker = "x")
        plt.scatter(y[:,0], y[:,1], color = "b", marker = "o")
        plt.xlim(-5,15)
        plt.ylim(-5,10)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.plot(x1, y1)
        plt.show()





def selfCreated3D():
    #Creating the 2D dataset
    n = int(input("Enter the number of datapoints for which you want to generate the data : "))
    np.random.seed(10)
    x = np.random.multivariate_normal((2,7,2), [[1,0,0],[0,1,0],[0,0,1]], int(n/2))
    x1 = np.c_[x,np.ones(int(n/2))]
    y = np.random.multivariate_normal((9,12,15), [[1,0,0],[0,1,0],[0,0,1]], int(n/2))
    y1 = np.c_[y,np.ones(int(n/2))*-1]
    whole_dataset = np.r_[x1,y1]
    np.random.shuffle(whole_dataset)


    showLearning = input("Press 1 if you want to visualise the learning process otherwise press 0: ")
    while True:
        if showLearning == "1":
            showLearning = True
            break
        elif showLearning == "0":
            showLearning = False
            break
        else:
            print("Enter correct input : ")

    #creating theb perceptron instance and training it
    pp1 = Perceptron(0.01, 100)
    pp1.train(whole_dataset, showLearning = showLearning)

    #Plotting the boundry during the learning
    if showLearning == True:
        for k in range(len(pp1.showLearning)):
            ax = plt.axes(projection = "3d")
            ax.scatter3D(x[:,0], x[:,1], x[:,2], color = "r", marker = "x")
            ax.scatter3D(y[:,0], y[:,1], y[:,2],color = "b", marker = "o")
            ax.plot_surface(pp1.showLearning[k][0], pp1.showLearning[k][1], pp1.showLearning[k][2])
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            ax.set_title("Training ")
            plt.show()
    else: 
        ax = plt.axes(projection = "3d")
        ax.scatter3D(x[:,0], x[:,1], x[:,2], color = "r", marker = "x")
        ax.scatter3D(y[:,0], y[:,1], y[:,2],color = "b", marker = "o")
        x = np.linspace(-5,20,100)
        y = np.linspace(-5,20,100)
        x , y = np.meshgrid(x,y)
        z = (-pp1.weights[0] * x - pp1.weights[1] * y - pp1.bias)/pp1.weights[2]
        ax.plot_surface(x, y, z)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.show()
    



def setosaVsVersicular():
    """
    This method loads the Iris dataset, extract the setosa and versicular objects, and trains a perceptron on the that
    """
    #Loading the Iris dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)
    #Extracting the relevant data
    df = df[:100]
    df[4].iloc[:50] = 1
    df[4].iloc[50:] = -1
    df = np.array(df)
    np.random.seed(3284)
    np.random.shuffle(df)
    #creating theb perceptron instance and training it
    pp1 = Perceptron(0.01, 100)
    pp1.train(df, showLearning=True)


def main():
    """This method ask the user about the dataset on which you want to train the perceptron and direct
    the program to that dataset"""
    while True:
        menu = input("\nEnter from the following choices :\n1. Press 1 train for self created 2D dataset\n2. Press 2 train for self created 3D dataset\n3. Press 3 train for self setosa vs versicular dataset\n4. Press Q to quit : ")
        if menu == "1":
            selfCreated2D()
        if menu == "2":
            selfCreated3D()
        if menu == "3":
            setosaVsVersicular()
        elif menu == "q" or menu == "Q":
            break

if __name__ == "__main__":
    main()

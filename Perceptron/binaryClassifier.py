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

    def fit(self, c1, c2):
        """Fit training data.

        Parameters:
        c1 :
          All the datapoints belonging to class taken as positive
        c2 :
          All the datapoints belonging to class taken to be negative

        Returns
        self : object

        """
        #Merging the two classes and separating the datapoints and their labels
        X, label = self.labelingdata(c1,c2)
        #initialisation of weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0


        for _ in range(self.iterations): #iterating through number of itertions
          for datapoint, y in zip(X, label): #iterating through each datapoint
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

    def labelingdata(self,data1, data2):
      """It labels the given two sets as 1 and -1, then combines the data, shuffle them, and
      return the separated data and labels in the corresponding sequense"""
      data1[:,4] = 1
      data2[:,4] = -1
      X = np.r_[data1,data2]
      np.random.shuffle(X) #shuffling the data
      return X[:,:4].astype(np.float), X[:,4].astype(np.float)

    def evaluation(self, c1, c2):
      """This method takes the two different classes as testing data, labels them, takes its 
      prediction with trained parameters, check the prediction against actual label, count the
      true preictions and then return the accuracy of the model"""
      correctPrediction = 0
      X, y = self.labelingdata(c1, c2)
      for xi, target in zip(X, y):
        if self.predict(xi) == target:
          correctPrediction += 1
      return str((correctPrediction/X.shape[0]) * 100)

def dataLoader(file):
  """This methos takes the csv file return the data in the form of nd array"""
  data = []
  with open(file) as file:
      for line in file:
          data.append(line.strip().split(","))
      file.close()
  return np.array(data)

def dataSpliter(data):
  """As the data is in sequence with equal number entries for each class, so this method takes
  the data, split, and return the separated data for each class"""
  step = int(data.shape[0]/3)
  return data[:step,:], data[step:step * 2,:], data[2 * step:,:]

def main():
  np.random.seed(100)
  #Loading the data
  data = dataLoader("train.data")
  #Splitting of data in three classes
  class1, class2, class3 = dataSpliter(data)
  #Initialisation and fitting of models
  ppn1 = Perceptron()
  ppn1.fit(class1,class2)
  ppn2 = Perceptron()
  ppn2.fit(class2,class3)
  ppn3 = Perceptron()
  ppn3.fit(class1, class3)

  testData = dataLoader("test.data")
  #Splitting of data in three classes
  tclass1, tclass2, tclass3 = dataSpliter(testData)


  print("\n\t\t\t   Accuracies for Binary Classification")
  print("\nAccuracy Type\t\tClass1 & Class2\t\tClass2 & Class3\t\tClass1 & Class3\n")
  print("Training\t\t", ppn1.evaluation(class1, class2) + "%\t\t\t" + ppn2.evaluation(class2, class3) + "%\t\t\t" + ppn3.evaluation(class1, class3)+ "%")
  print("Testing\t\t\t", ppn1.evaluation(tclass1, tclass2) + "%\t\t\t" + ppn2.evaluation(tclass2, tclass3) + "%\t\t\t" + ppn3.evaluation(tclass1, tclass3)+ "%\n")


if __name__ == "__main__":
  main()

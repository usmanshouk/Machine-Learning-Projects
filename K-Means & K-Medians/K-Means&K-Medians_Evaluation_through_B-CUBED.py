#Usman Shoukat
#201537600
import numpy as np
from numpy.lib.function_base import median
import pandas as pd
import matplotlib.pyplot as plt
def dataLoader(file):
    """This method loads the data return it in the form of nd array"""
    data = []
    with open(file) as file:
        for line in file:
            data.append(line.strip().split(" "))
        file.close()
    return np.array(data)

def normalisation(dataset):
    """This method normalise the given data"""
    normalised = np.empty(dataset.shape)
    for ind in range(dataset.shape[0]):
        b = np.linalg.norm(dataset[ind])
        normalised[ind] = dataset[ind]/b

    return normalised

def euclideanDistance(x, y):
    "This method takes the two vectors and returns the euclidean distance between them"
    return np.linalg.norm(x-y)

def manhattanDistance(x, y):
    "This method takes the two vectors and returns the manhattan distance between them"
    return np.linalg.norm(x-y, ord=1)

def BCUBEDEvaluation(clusters, labels, animals, countries, fruits, veggies):
    """
    This method takes the clusters and labels of dataset and then returns the precision, recall, and fscore
    of the clusters.
    Parameters:
        Clusters : Clusters to be evaluated
        labels : Labels of whole data
        animals : labels of objects of animals class
        countries : labels of objects of countries class
        fruits : labels of objects of fruits class
        veggies : labels of objects of veggies class

    return:
        precision, recall, fscore
    
    
    """
    #initialising the dictionaries for each cluster
    counter = [{"animals":0, "countries": 0, "fruits": 0, "veggies": 0} for _ in range(len(clusters))]

    #Counting the objects of different classes in cluster
    for i in range(len(clusters)):
        for j in clusters[i]:
            if labels[j] in animals:
                counter[i]["animals"] += 1
            elif labels[j] in veggies:
                counter[i]["veggies"] += 1
            elif labels[j] in fruits:
                counter[i]["fruits"] += 1
            elif labels[j] in countries:
                counter[i]["countries"] += 1

    precision = 0
    recall = 0
    #computing the precision, recall and fscore
    for i in range(len(counter)):
        recall += (counter[i]["animals"] ** 2)/len(animals)
        recall += (counter[i]["veggies"] ** 2)/len(veggies)
        recall += (counter[i]["fruits"] ** 2)/len(fruits)
        recall += (counter[i]["countries"] ** 2)/len(countries)
        precision += (counter[i]["animals"] ** 2)/len(clusters[i])
        precision += (counter[i]["veggies"] ** 2)/len(clusters[i])
        precision += (counter[i]["fruits"] ** 2)/len(clusters[i])
        precision += (counter[i]["countries"] ** 2)/len(clusters[i])
    precision = round(precision/labels.shape[0], 3)
    recall = round(recall/labels.shape[0], 3)
    fscore = round((2*precision*recall) / (precision + recall), 3)
    return (precision, recall, fscore)

def kmeans(dataset, k):
    """
    This method takes the dataset and parameter k(number of clusters), cluster it by kmeans technique and
    returns the dataset in k clusters
    """
    np.random.seed(45)
    #initialising the means randomly
    means = list(np.random.choice(dataset.shape[0],k, replace=False))
    means = dataset[means]
    prevmean = []
    #running until converged
    while True:
        #initialising the k empty clusters
        clusters = [[] for i in range(k)]
        prevmean = means
        #iterting through each datapoint
        for i in range(dataset.shape[0]):
            min = 0
            for j in range(len(means)):
                #assigning datapoint to closest cluster
                if euclideanDistance(dataset[i], means[j]) < euclideanDistance(dataset[i], means[min]):
                    min = j
            clusters[min].append(i)
        #calculating the means (representatives of the clusters)
        means = [sum(dataset[x])/len(dataset[x]) for x in clusters]

        #Break if converged
        if all([all(a == b) for a,b in zip(prevmean, means)]):
            break
    
    return clusters

def kmedians(dataset, k):
    """
    This method takes the dataset and parameter k(number of clusters), cluster it by kmedians technique and
    returns the dataset in k clusters
    """
    np.random.seed(10)
    #initialising the medians randomly
    medians = list(np.random.choice(dataset.shape[0],k, replace=False))
    medians = dataset[medians]
    prevmedians = []
    #running until converged
    while True:
        #initialising the k empty clusters
        clusters = [[] for i in range(k)]
        prevmedians = medians
        #iterting through each datapoint
        for i in range(dataset.shape[0]):
            min = 0
            for j in range(len(medians)):
                #assigning datapoint to closest cluster
                if manhattanDistance(dataset[i], medians[j]) < manhattanDistance(dataset[i], medians[min]):
                    min = j
            clusters[min].append(i)

        #calculating the medians (representatives of the clusters)
        medians = [np.median(dataset[x], axis=0) for x in clusters]
        #Break if converged
        if all([all(a == b) for a,b in zip(prevmedians, medians)]):
            break
        
    return clusters

def plot(precisions, recalls, fscores, title):
    "This method takes precisions, recalls, and fscores, and plots the graph"
    x = range(1,10)    
    plt.plot(x, precisions, label = "Precision")
    plt.plot(x, recalls, label = "Recall")
    plt.plot(x, fscores, label = "F-Score")
    plt.legend()
    plt.title(title)
    plt.xlabel("Number of Clustering")
    plt.ylabel("Scores")
    # plt.savefig(title)
    plt.show()

def main():
    #Loading the data
    data1 = dataLoader("fruits")
    fruits = data1[:,0]
    data2 = dataLoader("veggies")
    veggies = data2[:,0]
    data3 = dataLoader("animals")
    animals = data3[:,0]
    data4 = dataLoader("countries")
    countries = data4[:,0]
    data = np.r_[data1, data2, data3, data4]
    label = data[:,0]
    dataset = data[:,1:].astype(float)

    setOfKClusters = {}
    clusterEvaluations = {} 

    precisions = []
    recalls = []
    fscores = []


    #Asking user which criteria to choose
    print("Choose from the following options : \n")
    choice = input("Press 1 for clustering by k_means without Normalisation\nPress 2 for clustering by k_means with Normalisation \
    \nPress 3 for clustering by k_medians without Normalisation\nPress 4 for clustering by k_medians with Normalisation : ")


    for k in range(1, 10):
        print(k, "\n")
        if choice == "1":
            setOfKClusters[k] = kmeans(dataset, k) #calling kmeans without notmalisation
            title = "K-Means without Normalised Data"
            
        elif choice == "2":
            setOfKClusters[k] = kmeans(normalisation(dataset), k) #calling kmeans with normalisation
            title = "K-Means with Normalised Data"

        elif choice == "3":
            setOfKClusters[k] = kmedians(dataset, k) #calling kmedians without notmalisation
            title = "K-Medians without Normalised Data"
            
        elif choice == "4":
            setOfKClusters[k] = kmedians(normalisation(dataset), k) #calling kmedians with normalisation
            title = "K-Medians with Normalised Data"
        
        #Evaluating the clusters
        clusterEvaluations[k] = BCUBEDEvaluation(setOfKClusters[k], label, animals, countries, fruits, veggies)

        precisions.append(clusterEvaluations[k][0])
        recalls.append(clusterEvaluations[k][1])
        fscores.append(clusterEvaluations[k][2])

    print("\n\tClustering Evaluation\n")
    print("-"*50)
    print("k\tPrecision\tRecall\t\tF-score")
    print("-"*50)
    for key, val in clusterEvaluations.items():
        # print("Number of Clusters : ", key)
        print(key,"\t", val[0], "\t\t", val[1], "\t\t", val[2])

    #plotting the precisions, recalls, and fscores for different number of clustering
    plot(precisions, recalls, fscores, title)




        
if __name__ == "__main__":
    main()
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch import optim
import torchvision.transforms as T
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def main():
    seed = 1000
    th.manual_seed(seed)
    if th.cuda.is_available():
    # Make CuDNN Determinist
        th.backends.cudnn.deterministic = True
        th.cuda.manual_seed(seed)
    #Directories
    trainDirectory = "C:/Users/ushou/Downloads/Data Science and AI/Applied AI/Assignments/Assignment 3/covid19_dataset_32_32/train"
    testDirectory = "C:/Users/ushou/Downloads/Data Science and AI/Applied AI/Assignments/Assignment 3/covid19_dataset_32_32/test"

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    #checking for model
    net = input("Press 1 if you want to use simple net\nPress any other key if you want to use modified net : ")

    if net == "1":
        #checking for greyScale and Loading the images
        greyScale = input("Enter 1 if you want to use convert channel of image to 1 otherwise press any other key : ")
        if greyScale == "1":
            greyScale = True
        else:
            greyScale = False
    
        trainLoader, valLoader, testLoader = imageDataLoader(trainDirectory, testDirectory, greyScale)

        #Initialising a network
        simpleNet = SimpleConvolutionalNetwork(greyScale)
        #Training the model
        trainHistory, valHistory = train(simpleNet, batchSize=2, nEpochs=30, learningRate=0.001, device=device, trainSet=trainLoader, valSet=valLoader)

        #plotting the curves
        plotLosses(trainHistory, valHistory, "Simple Net")

        #computing the accuracies
        print("\nComputing accuracy...")
        test(simpleNet, trainLoader, valLoader, testLoader, device=device)


    else:
        #checking for greyScale and Loading the images
        greyScale = input("Enter 1 if you want to use greyScale image otherwise press any other key : ")
        if greyScale == "1":
            greyScale = True
        else:
            greyScale = False
        trainLoader, valLoader, testLoader = imageDataLoader(trainDirectory, testDirectory, greyScale, modifiedModel = True)

        #Initialising a network
        modifiedNet = ModifiedConvolutionalNetwork(greyScale)

        #Training the model
        trainHistory, valHistory = train(modifiedNet, batchSize=2, nEpochs=30, learningRate=0.001,  device=device, trainSet=trainLoader, valSet=valLoader)

        #plotting the curves
        plotLosses(trainHistory, valHistory, "Modified Net")

        #computing the accuracies
        print("\nComputing accuracy...")
        test(modifiedNet, trainLoader, valLoader, testLoader,  device=device)


    
def imageDataLoader(trainDirectory, testDirectory, greyScale = False, modifiedModel = False):
    """
    This method takes the directory of test and train data, split the train data in training and validation data,
    make the apprpriate transformations and return the training, testing and valdation data
    """
    #Making Transforms
    if modifiedModel:
        if greyScale:
            transform = T.Compose([
                T.Grayscale(num_output_channels=1), 
                T.ToTensor(), T.RandomHorizontalFlip()])
        else:
            transform = T.Compose([
                T.ToTensor(), T.RandomHorizontalFlip()])
    else:
        if greyScale:
            transform = T.Compose([
                T.Grayscale(num_output_channels=1), 
                T.ToTensor()])
        else:
            transform = T.Compose([
                T.ToTensor()])

    #Loading the images from firectory
    dataset =   ImageFolder(trainDirectory, transform=transform)
    testData = ImageFolder(testDirectory, transform=transform)
    batchSize = 2
    #Defining the spliting
    valSize = int(len(dataset) * 0.2)
    trainSize =  len(dataset) - valSize
    valSize = int(len(dataset) * 0.2)
    trainData, valData = random_split(dataset, [trainSize, valSize])

    #Loading the data in batches
    trainLoader = DataLoader(trainData, batchSize, shuffle=True)
    testLoader = DataLoader(testData, batchSize, shuffle=True)
    valLoader = DataLoader(valData, batchSize, shuffle=True)


    return trainLoader, valLoader, testLoader

class SimpleConvolutionalNetwork(nn.Module):
    """
    A simple neural network with one convolutional layer, one pooling layer and two fully connected layers
    
    """
    def __init__(self, greyScale):
        super(SimpleConvolutionalNetwork, self).__init__()
        #deciding the input channels
        if greyScale:
            inChannels = 1
        else:
            inChannels = 3

        #defining hyperparameters
        kernelSize = 3
        stride =  1
        self.outChannels = 32
        zeroPadding = 1
        inSize = 32
        kernelSizePool = 2
        stridePool =  2
        zeroPaddingPool = 0

        #Initialising the layers
        self.conv1 = nn.Conv2d(inChannels, self.outChannels, kernel_size=kernelSize, stride=stride, padding=zeroPadding)
        #Computing the out height
        h2 = outSize(inSize = inSize, kernelSize = kernelSize, stride=stride, padding=zeroPadding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #Computing the out height
        self.h3 = outSize(inSize = h2, kernelSize = kernelSizePool, stride=stridePool, padding=zeroPaddingPool)
        self.fc1 = nn.Linear(int(self.outChannels * self.h3 * self.h3), 64) 
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        """
        Forward pass of the network
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        #Flattening the convolution
        x = x.view(-1, int(self.outChannels * self.h3 * self.h3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModifiedConvolutionalNetwork(nn.Module):
    """
    A modified neural network with two convolutional layer, two batchnorm layers, one pooling layer and three fully connected layers
    
    """
    def __init__(self, greyScale):
        super(ModifiedConvolutionalNetwork, self).__init__()
        #deciding the input channels
        if greyScale:
            inChannels = 1
        else:
            inChannels = 3

        #defining hyperparameters
        kernelSize = 3
        stride =  1
        self.outChannels = 64
        zeroPadding = 1
        inSize = 32
        kernelSizePool = 2
        stridePool =  2
        zeroPaddingPool = 0

        #Initialising the layers
        self.conv1 = nn.Conv2d(inChannels, 32, kernel_size=kernelSize, stride=stride, padding=zeroPadding)
        #Computing the out height
        h2 = outSize(inSize = inSize, kernelSize = 3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernelSize, stride=stride, padding=zeroPadding)
        #Computing the out height
        h3 = outSize(inSize = h2, kernelSize = 3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #Computing the out height
        self.h4 = outSize(inSize = h3, kernelSize = kernelSizePool, stride=stridePool, padding=zeroPaddingPool)
        self.fc1 = nn.Linear(int(self.outChannels * self.h4 * self.h4), 32)  
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        """
        Forward pass of the network
        """
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool(x)
        #Flattening the convolution
        x = x.view(-1, int(self.outChannels * self.h4 * self.h4))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def outSize(inSize, kernelSize, stride=1, padding=0):
    """This method Calculats the out size of layer by using the formula (N -F+2P)/S+1"""
    return int((inSize - kernelSize + 2 * padding) / stride) + 1

def createLossAndOptimizer(net, learningRate=0.001):
    """
    This method defined the loss criteria and the optimiser
    """
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    return criterion, optimizer



def train(net, batchSize, nEpochs, learningRate, trainSet, valSet, device):
    """
    Train a neural network and prints out its evolution during the training

    Parameters:    
        net: Convolutional neural network
        batchSize: size of  the batch
        nEpochs: (int)  Number of iterations on the training set
        learningRate: (float) learning rate used by the optimizer
    """
    print("\n===== HYPERPARAMETERS =====")
    print("Batch Size = ", batchSize)
    print("number of Epochs = ", nEpochs)
    print("Learning Rate = ", learningRate)
    print("=" * 27)

    #defining the loss criteria and optimiser
    criterion, optimizer = createLossAndOptimizer(net, learningRate)
    # initialisation of variables used to track record of loss
    trainHistory = []
    valHistory = []

    #Track of best error to update the model
    bestError = np.inf
    bestModelPath = "best_model.pth"
    
    # Move model to gpu if possible
    net = net.to(device)

    for epoch in range(nEpochs):  # loop over the dataset multiple times

        runningLoss = 0.0
        # print_every = nMinibatches // 10
        # start_time = time.time()
        totalTrainLoss = 0
        

        
        for inputs, labels in trainSet:

            # Move tensors to correct device
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            runningLoss += loss.item()
            totalTrainLoss += loss.item()


        trainHistory.append(totalTrainLoss / len(trainSet))

        totalValLoss = 0


        # Do a pass on the validation set
        # We don't need to compute gradient,
        # we save memory and computation using th.no_grad()
        with th.no_grad():
          for inputs, labels in valSet:
              # Move tensors to correct device
              inputs, labels = inputs.to(device), labels.to(device)
              # Forward pass
              predictions = net(inputs)
              valLoss = criterion(predictions, labels)
              totalValLoss += valLoss.item()
            
        valHistory.append(totalValLoss / len(valSet))

        # Save model that performs best on validation set
        if totalValLoss < bestError:
            bestError = totalValLoss
            th.save(net.state_dict(), bestModelPath)

        print("Epoch = ",epoch + 1, ", Training loss = {:.2f}".format(totalTrainLoss / len(trainSet)), ", Validation loss = {:.2f}".format(totalValLoss / len(valSet)))

    
    # Load best model
    net.load_state_dict(th.load(bestModelPath))
    
    return trainHistory, valHistory


def datasetAccuracy(net, dataLoader, device, name=""):
    """
    This method takes the network and data compute the accuracy of model on the data
    """
    correct = 0
    total = 0
    for images, labels in dataLoader:

        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = th.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * float(correct) / total
    print('Accuracy of the network on the {} {} images: {:.2f} %'.format(total, name, accuracy))


def test(net, trainLoader, valLoader, testLoader, device):
    """
    This method takes the training, validation and test data, calls the accuracy function to calculate accuracy of each data on the traind model
    """
    datasetAccuracy(net, trainLoader, name = "Train",  device = device)
    datasetAccuracy(net, valLoader, name = "Validation",  device = device)
    datasetAccuracy(net, testLoader, name = "Test",  device = device,)
    


def plotLosses(trainHistory, valHistory, title):
    """
    This method takes in the train history and the validation history on plot it
    """

    x = np.arange(1, len(trainHistory) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, trainHistory, label="Training loss", linewidth=2)
    plt.plot(x, valHistory, label="Validation loss", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the training and validation loss for " + title)
    plt.show()



if __name__ == "__main__":
    main()
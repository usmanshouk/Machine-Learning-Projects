Requirements: Python 3.0 or later
Libraries used: numpy, cv2, torch, torch, matplotlib, tqdm

To run this code, user must have files covid19_dataset_32_32.zip and covid19_dataset_800_800.zip on your local machine.

The code is written to train a Convolutional Neural Network to classify Covid-19 related images.
More specifically, the dataset consists three types of CT Scan images (a) Covid; (b) Normal and (c) Viral Pneumonia.

Code ask the user to choose whether to train on simpler or modified model and whether you want 
to change the channel to one(Grey scale) or not.

Accordingly it trains the model and plots the training and validation loss curves and also prints out the accuracy
of the model on training, validation and testing set. 